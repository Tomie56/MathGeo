"""
启动脚本（兼容bash自动化调用）:
python ./scripts/call_api/get_question.py <输入jsonl路径> --output <输出jsonl路径>

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
from tqdm import tqdm
import requests
from mimetypes import guess_type
import argparse  # 新增：解析命令行参数

def log_message(message):
    """增强型日志记录"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

ips = [
    '10.119.25.133'
]
URLS = [
    f"http://{ip}:8000" for ip in ips
]
MAX_CONCURRENT_PER_SERVER = 80  # 每个ip的最大并发请求数
max_retry = 3   # 每个请求的最大重试次数
req_per_question = 1    # 一条数据跑多少次
server_status = defaultdict(lambda: {"pending": 0, "response_time": 1.0})

# 修正AOSS客户端配置路径（适配当前用户目录）
_aoss_client = client.Client('/mnt/afs/jingjinhao/aoss.conf')

SYSTEM_PROMPT = "<|im_start|>system\nDescribe the picture in detail.<|im_end|>\n"
USER_START = "<|im_start|>user\n"
ASSISTANT_START = "<|im_start|>assistant\n"
IM_END = "<|im_end|>\n"
IMG_TAG = "<img></img>\n"
AUDIO_TAG = "<audio> ></audio>\n"

class APIOptimizer:
    def __init__(self):
        self.request_queue = asyncio.Queue()
        self.progress = {}
        self.server_cycle = itertools.cycle(URLS)
        self.last_log_time = datetime.now()
        self.start_time = datetime.now()
        self.success_count = 0
        self.failure_count = 0
        self.total_processed = 0  # 总处理数统计

    async def encode_image_to_base64(self, image_path):
        if 's3' in image_path:
            return base64.b64encode(_aoss_client.get(image_path)).decode("utf-8")
        else:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

    async def local_image_to_data_url(self, image_path):
        # 推测图片MIME类型
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            if image_path.split('.')[-1] == 'webp':
                mime_type = 'image/webp'
            else:
                mime_type = 'application/octet-stream'

        # 编码为base64
        base64_encoded_data = await self.encode_image_to_base64(image_path)
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

    async def construct_messages(self, image_path, item):
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
                        'text': f'Based on the provided text and image information, generate a math problem stem for me. The stem should adhere to the language style of common problem formats, fully, rigorously, and concisely integrating and describing the image information. It should then pose a question formatted according to the content in Question Reference, ensuring the question is accurate and written in English.\n\n'
                                f'Question Reference: {item["question"]}\n'
                                f'Description of Geometric Context: {item["qt_description"]}\n'
                                f'Full Geometric Description: {item["description"]}'
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
                        "model": "Qwen3-VL-235B-A22B-Instruct",
                        "messages": await self.construct_messages(image_path, item),
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

    async def worker(self, session, pbar, tmp_dir):
        """增强进度跟踪的工作线程"""
        while True:
            task_retrieved = False
            try:
                # 从队列获取任务（idx：样本序号，item：原始数据）
                idx, item = await self.request_queue.get()
                task_retrieved = True
                
                # 中间文件路径（tmp_dir下按序号命名）
                output_json = os.path.join(tmp_dir, f"{idx}.json")
                start_time = datetime.now()
                status = "失败"
                retry_history = []

                # 跳过已存在且结果完整的文件
                if os.path.exists(output_json):
                    try:
                        async with aiofiles.open(output_json, 'r') as f:
                            content = await f.read()
                            existing_data = json.loads(content)
                            if len(existing_data.get("generated_question", [])) >= req_per_question:
                                self.total_processed += 1
                                pbar.update(1)
                                self.request_queue.task_done()
                                continue
                    except Exception as e:
                        log_message(f"文件校验异常 | 样本{idx} | 错误:{str(e)}")

                # 处理图片路径（兼容单图/多图）
                image_path = []
                if isinstance(item['image'], list):
                    image_path = item['image']  # 输入jsonl的image已为绝对路径（来自format.py）
                else:
                    image_path = [item['image']]

                # API调用获取结果
                results = []
                for attempt in range(req_per_question + max_retry):
                    result = await self.call_openai_api_async(session, idx, image_path, item)
                    if result:
                        results.append(result)
                        if len(results) >= req_per_question:
                            break
                    retry_history.append(f"第{attempt+1}次:{'成功' if result else '失败'}")

                # 写入中间文件（原子操作保障完整性）
                tmp_path = f"{output_json}.tmp"
                async with aiofiles.open(tmp_path, 'w', encoding='utf-8') as f:
                    # 新增generated_question字段存储生成结果，保留原始字段
                    item['generated_question'] = results[:req_per_question]
                    await f.write(json.dumps(item, ensure_ascii=False, indent=2))
                os.replace(tmp_path, output_json)

                # 统计状态
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
                
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({
                    '成功': self.success_count,
                    '失败': self.failure_count,
                    '速度': f"{(self.total_processed)/(datetime.now()-self.start_time).total_seconds():.2f}条/s",
                    '当前负载': '/'.join(f'{s["pending"]}' for s in server_status.values()),
                    '累计处理': self.total_processed
                })

            except asyncio.CancelledError:
                log_message(f"工作线程被正常取消 | 样本{idx if task_retrieved else '未知'} | 已处理任务数:{self.total_processed}")
                break
            except Exception as e:
                log_message(f"工作线程业务异常 | 样本{idx if task_retrieved else '未知'} | 错误:{str(e)}")
                traceback.print_exc()
            finally:
                if task_retrieved:
                    self.request_queue.task_done()

    async def process_jsonl(self, input_path, output_path):
        """核心逻辑：读取输入jsonl，处理后输出到指定路径"""
        # 1. 读取输入jsonl，过滤无效条目
        log_message(f"开始读取输入文件：{input_path}")
        items = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # 必要字段检查
                    required_fields = ['image', 'question', 'qt_description', 'description', 'gt']
                    if all(field in data for field in required_fields):
                        items.append(data)
                    else:
                        log_message(f"跳过无效条目（行号{line_num}）：缺少必要字段")
                except json.JSONDecodeError:
                    log_message(f"跳过无效条目（行号{line_num}）：JSON解析失败")
        
        total_items = len(items)
        log_message(f"输入文件读取完成 | 有效条目数：{total_items}")
        if total_items == 0:
            log_message("无有效条目可处理，退出程序")
            return

        # 2. 创建临时目录（存储中间文件）
        tmp_dir = os.path.join(os.path.dirname(output_path), "tmp_question")
        os.makedirs(tmp_dir, exist_ok=True)
        log_message(f"中间文件存储目录：{tmp_dir}")

        # 3. 填充请求队列
        for idx, item in enumerate(items):
            await self.request_queue.put((idx, item))

        # 4. 启动工作线程处理任务
        connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_PER_SERVER*len(URLS))
        async with aiohttp.ClientSession(connector=connector) as session:
            with tqdm(total=total_items, desc="生成问题", file=sys.stdout, 
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [剩余{remaining}] {postfix}") as pbar:
                # 启动工作线程（数量=并发数上限或条目数，取较小值）
                worker_count = min(MAX_CONCURRENT_PER_SERVER*len(URLS), total_items)
                workers = [asyncio.create_task(self.worker(session, pbar, tmp_dir)) for _ in range(worker_count)]
                
                # 等待所有任务完成
                await self.request_queue.join()
                # 取消所有工作线程
                for w in workers:
                    if not w.done():
                        w.cancel()
                await asyncio.gather(*workers, return_exceptions=True)

        # 5. 汇总中间文件到最终输出jsonl
        log_message(f"开始汇总结果到：{output_path}")
        success_count = 0
        async with aiofiles.open(output_path, 'w', encoding='utf-8') as outf:
            for idx in range(total_items):
                json_file = os.path.join(tmp_dir, f"{idx}.json")
                if os.path.exists(json_file):
                    try:
                        async with aiofiles.open(json_file, 'r') as f:
                            item = json.loads(await f.read())
                            # 只保留生成成功的条目
                            if len(item.get("generated_question", [])) >= req_per_question:
                                await outf.write(json.dumps(item, ensure_ascii=False) + '\n')
                                success_count += 1
                    except Exception as e:
                        log_message(f"跳过异常中间文件（序号{idx}）：{str(e)}")

        # 6. 输出最终统计
        log_message(f"结果汇总完成 | 输出路径：{output_path}")
        log_message(f"最终统计：总条目{total_items} | 成功生成{success_count} | 成功率：{success_count/total_items:.1%}")

async def main_async():
    # 解析命令行参数（适配bash脚本调用）
    parser = argparse.ArgumentParser(description="调用API生成数学问题（基于图片和文本信息）")
    parser.add_argument("input_path", help="输入JSONL文件路径（来自format.py的输出）")
    parser.add_argument("--output", required=True, help="输出JSONL文件路径（存储生成的问题）")
    args = parser.parse_args()

    try:
        # 验证输入文件存在
        if not os.path.exists(args.input_path):
            log_message(f"错误：输入文件不存在 -> {args.input_path}")
            sys.exit(1)
        
        # 创建输出目录（如果不存在）
        output_dir = os.path.dirname(args.output)
        os.makedirs(output_dir, exist_ok=True)

        # 启动处理流程
        optimizer = APIOptimizer()
        await optimizer.process_jsonl(args.input_path, args.output)

    except Exception as e:
        log_message(f"全局异常 | 类型:{type(e).__name__} | 错误:{str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main_async())