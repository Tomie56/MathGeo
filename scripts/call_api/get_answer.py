"""
启动脚本:
nohup python -u /mnt/afs/liangjinwei/project/rl/code/rollout_construct.py > run.log 2>&1 &

python -u /mnt/afs/liangjinwei/project/rl/code/rollout_construct.py >/mnt/afs/liangjinwei/project/reasoning/code/run.log 2>&1

增强日志功能:
1. 实时显示每个样本的处理状态和详细错误信息
2. 进度条显示增强:
   - 当前处理速度
   - 预计剩余时间
   - 服务器负载状态
3. 错误处理增强:
   - 区分网络错误和API错误
   - 记录具体错误原因和发生位置
"""

import asyncio
import aiohttp
import json
import os
import itertools
import aiofiles
from collections import defaultdict
import sys
from datetime import datetime
import traceback
import base64
from aoss_client import client
from multiprocessing import Process
from tqdm import tqdm
import requests
from mimetypes import guess_type

def log_message(message):
    """增强型日志记录"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

ips = [
    '10.119.29.181'
]
URLS = [
    f"http://{ip}:8000" for ip in ips
]
MAX_CONCURRENT_PER_SERVER = 80  # 每个ip的最大并发请求数
max_retry = 3   # 每个请求的最大重试次数
req_per_question = 1    # 一条数据跑多少次
server_status = defaultdict(lambda: {"pending": 0, "response_time": 1.0})

_aoss_client = client.Client('/mnt/afs/liangjinwei/aoss.conf')

SYSTEM_PROMPT = "<|im_start|>system\nDescribe the picture in detail.<|im_end|>\n"
USER_START = "<|im_start|>user\n"
ASSISTANT_START = "<|im_start|>assistant\n"
IM_END = "<|im_end|>\n"
IMG_TAG = "<img></img>\n"
AUDIO_TAG = "<audio></audio>\n"

class APIOptimizer:
    def __init__(self):
        self.request_queue = asyncio.Queue()
        self.progress = {}
        self.server_cycle = itertools.cycle(URLS)
        self.last_log_time = datetime.now()
        self.start_time = datetime.now()
        self.success_count = 0
        self.failure_count = 0
        self.total_processed = 0  # 新增总处理数统计

    async def encode_image_to_base64(self, image_path):
        if 's3' in image_path:
            # return get_ceph_img(image_path)
            # return base64.b64encode(io.BytesIO(client.get(image_path))).decode("utf-8")
            return base64.b64encode(_aoss_client.get(image_path)).decode("utf-8")
        else:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

    async def local_image_to_data_url(self, image_path):
        # Guess the MIME type of the image based on the file extension
        mime_type, _ = guess_type(image_path)
        # mime_type = imghdr.what(image_path)
    
        if mime_type is None:
            if image_path.split('.')[-1] =='webp':
                mime_type = 'image/webp'
            else:
                mime_type = 'application/octet-stream'  # Default MIME type if none is found

        # Read and encode the image file
        base64_encoded_data = await self.encode_image_to_base64(image_path)

        # Construct the data URL
        return f"data:{mime_type};base64,{base64_encoded_data}"

    async def dynamic_load_balancer(self):
        """智能负载均衡策略"""
        try:
            candidates = [url for url in URLS if server_status[url]["pending"] < MAX_CONCURRENT_PER_SERVER*0.8]
            if candidates:
                return min(candidates, key=lambda x: server_status[x]["pending"]*server_status[x]["response_time"])
            return next(self.server_cycle)
        except Exception as e:
            log_message(f"负载均衡异常 | 错误:{str(e)}")
            return URLS[0]


    async def construct_messages(self, image_path, item):  # 修改
        prompt = "Solve the following problem step by step. Show your full thinking process clearly — including any formulas, substitutions, or logical deductions you make. Do not skip steps. At the end of your response, place the final numerical answer inside a LaTeX box using \\boxed{}\n\n"
        
        messages = [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'image_url',
                        'image_url': {'url': await self.local_image_to_data_url(image)}
                    } for image in image_path
                ] + [
                    {
                        'type': 'text',
                        'text': prompt +
                                f'Question: {item["question"][0]}'
                    }
                ]
            }
        ]
        return messages

    async def call_openai_api_async(self, session, idx, image_path, item):
        """增强错误处理的API调用"""
        base_url = None
        for retry in range(max_retry):
            try:
                base_url = await self.dynamic_load_balancer()
                server_status[base_url]["pending"] += 1
                start_time = datetime.now()

                async with session.post(
                    f"{base_url}/v1/chat/completions",
                    headers={"Authorization": "Bearer EMPTY"},
                    json={
                        "model": "Qwen3-VL-235B-A22B-Instruct",#修改
                        "messages": await self.construct_messages(image_path, item)#修改,

                    },
                    timeout=aiohttp.ClientTimeout(total=2000)
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        rt = (datetime.now() - start_time).total_seconds()
                        server_status[base_url]["response_time"] = 0.9 * server_status[base_url]["response_time"] + 0.1 * rt
                        res = result['choices'][0]['message']['content']
                        return res
                    else:
                        response_text = await response.text()
                        log_message(f"API异常 | 样本{idx} | 状态码:{response.status} | URL:{base_url} | 响应:{response_text[:200]}")
                        
            except (aiohttp.ClientConnectorError, aiohttp.ServerTimeoutError, asyncio.TimeoutError) as e:
                error_type = type(e).__name__
                log_message(f"网络异常 | 样本{idx} | 类型:{error_type} | URL:{base_url or '未知'} | 错误:{str(e)}")
            except json.JSONDecodeError as e:
                log_message(f"JSON解析失败 | 样本{idx} | URL:{base_url or '未知'} | 错误:{str(e)}")
            except KeyError as e:
                log_message(f"响应格式错误 | 样本{idx} | URL:{base_url or '未知'} | 缺失字段:{str(e)}")
            except Exception as e:
                log_message(f"未处理异常 | 样本{idx} | 类型:{type(e).__name__} | 错误:{str(e)}")
                traceback.print_exc()
            finally:
                if base_url:
                    server_status[base_url]["pending"] -= 1
                await asyncio.sleep((2 + retry)**2)
        return None

    async def worker(self, session, pbar):
        """增强进度跟踪的工作线程（修复task_done计数异常 + 保障JSONL数据完整性）"""
        while True:
            task_retrieved = False  # 标记：是否成功从队列获取任务（核心修复点1）
            try:
                # 1. 从队列获取任务（仅此时标记为“已获取”，避免取消时误调用task_done）
                idx, item, image_path, dataset_dir = await self.request_queue.get()
                task_retrieved = True  # 走到此处说明任务获取成功，后续需对应task_done
                
                output_json = os.path.join(dataset_dir, f"{idx}.json")
                start_time = datetime.now()
                status = "失败"
                retry_history = []

                try:
                    # 2. 处理“文件已存在”场景（补充task_done，避免跳过计数）
                    if os.path.exists(output_json):
                        log_message(f"文件已存在 | 样本{idx} | 路径:{output_json}")  # 替换print为结构化日志
                        async with aiofiles.open(output_json, 'r') as f:
                            content = await f.read()
                            existing_data = json.loads(content)
                            # 若已有结果满足要求，直接标记进度并释放任务（核心修复点2）
                            if len(existing_data["answer"]) >= req_per_question - 1:
                                self.total_processed += 1
                                pbar.update(1)
                                self.request_queue.task_done()  # 必须调用：已获取任务需释放队列计数
                                continue  # 跳过后续处理，直接进入下一轮循环
                except Exception as e:
                    log_message(f"文件校验异常 | 样本{idx} | 错误:{str(e)}")
                    # 校验异常不终止，继续尝试重新生成结果

                # 3. API调用重试逻辑（原有逻辑保留，确保结果完整性）
                results = []
                for attempt in range(req_per_question + max_retry):
                    result = await self.call_openai_api_async(session, idx, image_path, item)
                    if result:
                        results.append(result)
                        if len(results) >= req_per_question:
                            break  # 满足所需结果数量，提前终止重试
                    retry_history.append(f"第{attempt+1}次:{'成功' if result else '失败'}")

                # 4. 写入结果文件（临时文件→正式文件，避免JSONL汇总时读取不完整文件）
                tmp_path = f"{output_json}.tmp"
                async with aiofiles.open(tmp_path, 'w', encoding='utf-8') as f:
                    item['answer'] = results[:req_per_question]  # 截取所需数量结果
                    await f.write(json.dumps(item, ensure_ascii=False, indent=2))
                os.replace(tmp_path, output_json)  # 原子替换，保障文件完整性（JSONL汇总依赖）

                # 5. 状态统计与日志输出
                elapsed = (datetime.now() - start_time).total_seconds()
                if len(results) >= req_per_question:
                    status = "成功"
                    self.success_count += 1
                elif len(results) > 0:
                    status = "部分成功"
                else:
                    status = "失败"
                    self.failure_count += 1
                
                self.total_processed += 1
                log_message(f"样本{idx} | 状态:{status} | 耗时:{elapsed:.2f}s | 重试记录:{' '.join(retry_history)} | 有效结果:{len(results)}/{req_per_question}")
                
                # 6. 更新进度条（含实时统计，便于监控JSONL最终生成规模）
                pbar.update(1)
                pbar.set_postfix({
                    '成功': self.success_count,
                    '失败': self.failure_count,
                    '速度': f"{(self.total_processed)/(datetime.now()-self.start_time).total_seconds():.2f}条/s",
                    '当前负载': '/'.join(f'{s["pending"]}' for s in server_status.values()),
                    '累计处理': self.total_processed  # 新增：直观显示已处理数量（JSONL汇总基础）
                })

            except asyncio.CancelledError:
                # 捕获“任务被取消”异常（如处理完所有任务后关闭worker）
                log_message(f"工作线程被正常取消 | 样本{idx if task_retrieved else '未知'} | 已处理任务数:{self.total_processed}")
                break  # 退出循环，终止worker
            except Exception as e:
                # 捕获其他业务异常（避免单个任务失败导致worker崩溃）
                log_message(f"工作线程业务异常 | 样本{idx if task_retrieved else '未知'} | 错误:{str(e)}")
                traceback.print_exc()
            finally:
                # 7. 仅当任务成功获取时，才释放队列计数（核心修复点3）
                if task_retrieved:
                    self.request_queue.task_done()

                
    async def process_all_datasets(self, json_file_path, output_dir):##修改，新定义
        with open(json_file_path, 'r') as f:
            all_datasets = json.load(f)
        base_name = os.path.splitext(os.path.basename(json_file_path))[0]
        tasks = []
        for key, data in all_datasets.items():
            task = self.process_dataset_async(key, data, output_dir)
            tasks.append(task)
        try:
            results = await asyncio.gather(*tasks)
        except Exception as e:
            log_message(f"数据集处理异常 | 错误:{str(e)}")
            raise
        final_data_dict = dict(results)
        return final_data_dict, base_name
    
    async def process_dataset_async(self, key, data, output_dir):#修改
        """增强进度显示的数据集处理"""
        log_message(f"▄■▄ 开始处理数据集: {key} ■▄■")
        dataset_dir = os.path.join(output_dir, key)
        os.makedirs(dataset_dir, exist_ok=True)

        with open(data['annotation'], 'r') as f:
            items = [json.loads(line.strip()) for line in f]
            total_items = len(items)
            for idx, item in enumerate(items):
                image_path = []
                if 'image' not in item:
                    log_message(f"样本{idx} | 无image字段 | 跳过处理")
                    continue
                if isinstance(item['image'], list):
                    for img in item['image']:
                        image_path.append(os.path.join(data['root'], img))
                else:
                    image_path.append(os.path.join(data['root'], item['image']))
                await self.request_queue.put((idx, item, image_path, dataset_dir))
    

        connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_PER_SERVER*len(URLS))
        
        async with aiohttp.ClientSession(connector=connector) as session:
            with tqdm(total=len(items), desc=f"处理 {key}", file=sys.stdout, 
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [剩余{remaining}] {postfix}") as pbar:
                workers = [asyncio.create_task(self.worker(session, pbar)) 
                          for _ in range(min(MAX_CONCURRENT_PER_SERVER*len(URLS), total_items))]
                
                await self.request_queue.join()
                for w in workers:
                    if not w.done():
                        w.cancel()

        final_jsonl = os.path.join(output_dir, f"{key}.jsonl")
        success_count = 0
        async with aiofiles.open(final_jsonl, 'w', encoding='utf-8') as outf:
            for json_file in os.listdir(dataset_dir):
                if json_file.endswith('.json'):
                    async with aiofiles.open(os.path.join(dataset_dir, json_file), 'r') as f:
                        item = json.loads(await f.read())
                        if len(item["answer"]) == req_per_question:
                            await outf.write(json.dumps(item, ensure_ascii=False) + '\n')
                            success_count += 1
        
        log_message(f"▀■▀ 完成数据集: {key} ■▀■ | 总数:{total_items} | 成功:{success_count} | 成功率:{success_count/total_items:.1%}")

        return key, {**data, "annotation": final_jsonl, "length": total_items}

async def main_async():#修改
    try:
        optimizer = APIOptimizer()
        json_file_path = "/mnt/afs/jingjinhao/project/GeoChain/MathGeo/scripts/call_api/answer.json"
        output_dir = '/mnt/afs/jingjinhao/project/GeoChain/MathGeo/scripts/call_api/'
        os.makedirs(output_dir, exist_ok=True)

        final_data_dict, base_name = await optimizer.process_all_datasets(json_file_path, output_dir)
        output_filename = f"{base_name}_v1.json"
        output_path = os.path.join(output_dir, output_filename)
        async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(final_data_dict, ensure_ascii=False, indent=4))


    except Exception as e:
        log_message(f"全局异常 | 类型:{type(e).__name__} | 错误:{str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main_async())