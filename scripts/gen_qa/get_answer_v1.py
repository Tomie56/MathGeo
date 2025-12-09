import asyncio
import json
import os
import itertools
import aiofiles
from collections import defaultdict
import sys
from datetime import datetime
import traceback
import base64
from tqdm import tqdm
from mimetypes import guess_type
import argparse
import re
from openai import AsyncOpenAI

def log_message(message):
    """增强型日志记录"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

# 环境变量/配置优先
API_KEY = "495e7f4ae82ddc5ccdb928b1bb686375"
BASE_URL = "https://dl.yunstorm.com/v1"
# 恢复用户指定的模型名称
MODEL = "qwen3-vl-235b-a22b-thinking"  

MAX_CONCURRENT = 80  # 最大并发数
max_retry = 3
server_status = defaultdict(lambda: {"pending": 0, "response_time": 1.0})

# AOSS客户端（按需保留）
try:
    from aoss_client import client
    _aoss_client = client.Client('/mnt/afs/jingjinhao/aoss.conf')
except Exception as e:
    log_message(f"AOSS客户端初始化警告：{str(e)} | 忽略即可")
    _aoss_client = None

class APIOptimizer:
    def __init__(self, req_per_question):
        self.request_queue = asyncio.Queue()
        self.start_time = datetime.now()
        self.success_count = 0
        self.failure_count = 0
        self.total_processed = 0
        self.req_per_question = req_per_question
        # 初始化异步客户端（对齐官方配置）
        self.client = AsyncOpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
        )

    async def encode_image_to_base64(self, image_path):
        """图片Base64编码（兼容本地/S3）"""
        if not os.path.exists(image_path) and 's3' not in image_path:
            log_message(f"图片路径无效：{image_path}")
            return None
        try:
            if 's3' in image_path and _aoss_client:
                return base64.b64encode(_aoss_client.get(image_path)).decode("utf-8")
            else:
                with open(image_path, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            log_message(f"图片编码失败：{image_path} | {str(e)}")
            return None

    async def local_image_to_data_url(self, image_path):
        """生成图片DataURL"""
        base64_data = await self.encode_image_to_base64(image_path)
        if not base64_data:
            return None
        
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            ext = image_path.split('.')[-1].lower()
            mime_map = {'webp': 'image/webp', 'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg'}
            mime_type = mime_map.get(ext, 'application/octet-stream')
        
        return f"data:{mime_type};base64,{base64_data}"

    async def construct_messages(self, image_paths, item):
        """构建API请求消息（对齐官方格式）"""
        core_question = item["generated_question"][0] if item.get("generated_question") else "请解答这道几何题，给出详细步骤并在最后用\\boxed{}包裹最终答案。"
        prompt = core_question

        # 构建图片列表
        image_urls = []
        for img_path in image_paths:
            data_url = await self.local_image_to_data_url(img_path)
            if data_url:
                image_urls.append({
                    'type': 'image_url',
                    'image_url': {'url': data_url}
                })
            else:
                log_message(f"跳过无效图片：{img_path}")

        if not image_urls:
            log_message("无有效图片，无法构建请求")
            return None

        messages = [
            {
                'role': 'user',
                'content': image_urls + [
                    {
                        'type': 'text',
                        'text': prompt
                    }
                ]
            }
        ]
        return messages

    def is_answer_valid(self, answer):
        """校验答案是否包含闭合的\\boxed{}"""
        if not answer:
            return False
        return bool(re.search(r'\\boxed\{[^}]+\}', answer))

    async def call_openai_api_async(self, idx, image_paths, item):
        """异步流式调用API，提取reasoning_content和content"""
        messages = await self.construct_messages(image_paths, item)
        if not messages:
            log_message(f"样本{idx} | 无效请求（无图片/参数错误）")
            return None, None

        for retry in range(max_retry):
            reasoning_content = ""  # 存储完整思考过程
            answer_content = ""     # 存储完整答案
            try:
                server_status[BASE_URL]["pending"] += 1
                start_time = datetime.now()

                # 流式调用API（核心：stream=True）
                response = await self.client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=32768,
                    timeout=60,
                    stream=True,  # 开启流式响应
                    # stream_options={"include_usage": True}  # 按需开启token统计
                )

                # 遍历流式chunk，提取思考/答案（对齐官方逻辑）
                async for chunk in response:
                    if not chunk.choices:
                        continue  # 跳过usage chunk（如需统计可在此处理）
                    delta = chunk.choices[0].delta
                    # 提取思考过程（reasoning_content）
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        reasoning_content += delta.reasoning_content
                    # 提取最终答案（content）
                    if delta.content:
                        answer_content += delta.content

                # 更新响应时间
                rt = (datetime.now() - start_time).total_seconds()
                server_status[BASE_URL]["response_time"] = 0.9 * server_status[BASE_URL]["response_time"] + 0.1 * rt

                # 校验结果有效性
                if not reasoning_content:
                    reasoning_content = answer_content  # 兼容无思考过程的情况
                if not answer_content:
                    log_message(f"样本{idx} | 重试{retry+1} | 无答案内容")
                    continue

                return reasoning_content, answer_content
            
            except Exception as e:
                log_message(f"样本{idx} | 未知异常（重试{retry+1}/{max_retry}）| {type(e).__name__}: {str(e)}")
            finally:
                server_status[BASE_URL]["pending"] -= 1
        
        # 所有重试失败
        return None, None

    async def worker(self, pbar, tmp_dir):
        """工作线程：处理单条任务，写入目标格式"""
        while True:
            task_retrieved = False
            try:
                idx, item = await self.request_queue.get()
                task_retrieved = True
                output_json = os.path.join(tmp_dir, f"{idx}.json")
                start_time = datetime.now()

                # ======================== 关键修改1：强化已完成任务校验逻辑 ========================
                skip_flag = False
                if os.path.exists(output_json):
                    try:
                        async with aiofiles.open(output_json, 'r', encoding='utf-8') as f:
                            file_content = await f.read()
                            if not file_content:  # 处理空文件
                                log_message(f"样本{idx} | 临时文件为空，重新生成")
                            else:
                                existing_data = json.loads(file_content)
                                # 校验字段存在性 + 有效答案数
                                generated_answer = existing_data.get("generated_answer", [])
                                valid_answers = [ans for ans in generated_answer if self.is_answer_valid(ans)]
                                if len(valid_answers) >= self.req_per_question:
                                    log_message(f"样本{idx} | 已完成（有效答案数{len(valid_answers)}≥{self.req_per_question}），跳过")
                                    skip_flag = True
                    except json.JSONDecodeError:
                        log_message(f"样本{idx} | 临时文件JSON解析失败，重新生成")
                    except Exception as e:
                        log_message(f"样本{idx} | 校验临时文件异常 | {str(e)} | 重新生成")
                
                # 跳过已完成任务：必须先更新统计+进度条，再释放队列
                if skip_flag:
                    self.total_processed += 1
                    self.success_count += 1  # 已完成任务计入成功
                    pbar.update(1)
                    pbar.set_postfix({
                        '成功': self.success_count,
                        '失败': self.failure_count,
                        '速度': f"{self.total_processed/(datetime.now()-self.start_time).total_seconds():.2f}条/s",
                        '负载': f"{server_status[BASE_URL]['pending']}",
                    })
                    self.request_queue.task_done()
                    continue  # 直接跳过后续处理

                # 处理图片路径
                image_paths = item['image'] if isinstance(item['image'], list) else [item['image']]
                image_paths = [os.path.abspath(path) for path in image_paths]

                # 调用API生成多组思考/答案
                generated_thinking = []
                generated_answer = []
                retry_history = []
                for attempt in range(self.req_per_question + max_retry):
                    thinking, answer = await self.call_openai_api_async(idx, image_paths, item)
                    if thinking and answer:
                        if self.is_answer_valid(answer):
                            generated_thinking.append(thinking)
                            generated_answer.append(answer)
                            retry_history.append(f"第{attempt+1}次：成功（有效）")
                            if len(generated_answer) >= self.req_per_question:
                                break
                        else:
                            retry_history.append(f"第{attempt+1}次：成功（无\\boxed{{}}）")
                    else:
                        retry_history.append(f"第{attempt+1}次：失败")

                # 构建目标格式的输出项
                output_item = {
                    "image": item["image"] if isinstance(item["image"], str) else item["image"][0],
                    "question": item["generated_question"][0] if item.get("generated_question") else "",
                    "qt_description": item.get("qt_description", ""),
                    "description": item.get("description", ""),
                    "gt": item.get("gt", {"expr": "", "latex": ""}),
                    "diff": item.get("diff", 0.0),
                    "generated_question": item.get("generated_question", []),
                    "generated_thinking": generated_thinking[:self.req_per_question],
                    "generated_answer": generated_answer[:self.req_per_question],
                    "answer_validity": [self.is_answer_valid(ans) for ans in generated_answer[:self.req_per_question]]
                }

                # 原子写入临时文件
                tmp_path = f"{output_json}.tmp"
                async with aiofiles.open(tmp_path, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(output_item, ensure_ascii=False, indent=2))
                try:
                    os.replace(tmp_path, output_json)
                except OSError:
                    import shutil
                    shutil.move(tmp_path, output_json)

                # 统计状态
                elapsed = (datetime.now() - start_time).total_seconds()
                if len(generated_answer) >= self.req_per_question:
                    status = "成功"
                    self.success_count += 1
                elif len(generated_answer) > 0:
                    status = "部分成功"
                else:
                    status = "失败"
                    self.failure_count += 1

                self.total_processed += 1
                log_message(f"样本{idx} | {status} | 耗时{elapsed:.2f}s | 重试：{' '.join(retry_history)} | 有效：{len(generated_answer)}/{self.req_per_question}")

                # 更新进度条
                pbar.set_postfix({
                    '成功': self.success_count,
                    '失败': self.failure_count,
                    '速度': f"{self.total_processed/(datetime.now()-self.start_time).total_seconds():.2f}条/s",
                    '负载': f"{server_status[BASE_URL]['pending']}",
                })
                pbar.update(1)

            except asyncio.CancelledError:
                log_message(f"工作线程取消 | 样本{idx if task_retrieved else '未知'}")
                break
            except Exception as e:
                log_message(f"样本{idx if task_retrieved else '未知'} | 工作线程异常 | {str(e)}")
                traceback.print_exc()
            finally:
                if task_retrieved and not skip_flag:  # 仅未跳过的任务需要释放
                    self.request_queue.task_done()

    async def process_jsonl(self, input_path, output_path, clean_tmp):
        """主处理流程：读取输入→分发任务→汇总输出"""
        # 读取输入JSONL
        log_message(f"读取输入：{input_path}")
        items = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if 'image' not in data or not data.get('generated_question'):
                        log_message(f"跳过行{line_num}：缺少image/generated_question")
                        continue
                    items.append(data)
                except json.JSONDecodeError:
                    log_message(f"跳过行{line_num}：JSON解析失败")

        total_items = len(items)
        if total_items == 0:
            log_message("无有效条目，退出")
            return
        log_message(f"有效条目：{total_items} | 每条目生成{self.req_per_question}个答案")

        # ======================== 关键修改2：固定临时目录路径（避免拼接错误） ========================
        tmp_dir = os.path.abspath(os.path.join(os.path.dirname(output_path), "tmp_answer"))
        os.makedirs(tmp_dir, exist_ok=True)
        log_message(f"临时文件目录：{tmp_dir}")

        # 填充任务队列
        for idx, item in enumerate(items):
            await self.request_queue.put((idx, item))

        # 启动工作线程
        with tqdm(total=total_items, desc="生成答案", file=sys.stdout) as pbar:
            worker_count = min(MAX_CONCURRENT, total_items)
            workers = [asyncio.create_task(self.worker(pbar, tmp_dir)) for _ in range(worker_count)]
            await self.request_queue.join()
            # 取消剩余线程
            for w in workers:
                if not w.done():
                    w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

        # 汇总结果到输出文件
        log_message(f"汇总结果到：{output_path}")
        success_count = 0
        async with aiofiles.open(output_path, 'w', encoding='utf-8') as outf:
            for idx in range(total_items):
                json_file = os.path.join(tmp_dir, f"{idx}.json")
                if os.path.exists(json_file):
                    try:
                        async with aiofiles.open(json_file, 'r', encoding='utf-8') as f:
                            item = json.loads(await f.read())
                            if len(item.get("generated_answer", [])) >= self.req_per_question:
                                await outf.write(json.dumps(item, ensure_ascii=False) + '\n')
                                success_count += 1
                    except Exception as e:
                        log_message(f"跳过异常文件{idx} | {str(e)}")

        # 清理临时文件
        if clean_tmp:
            import shutil
            try:
                shutil.rmtree(tmp_dir)
                log_message(f"清理临时目录：{tmp_dir}")
            except Exception as e:
                log_message(f"清理失败 | {str(e)}")

        # 输出统计
        log_message(f"===== 处理完成 =====")
        log_message(f"总条目：{total_items} | 成功：{success_count} | 成功率：{success_count/total_items:.1%}")
        log_message(f"输出文件：{output_path}")

async def main_async():
    """命令行参数解析+主流程启动"""
    parser = argparse.ArgumentParser(description="流式提取思考过程+答案（对齐官方示例）")
    parser.add_argument("input_path", help="输入JSONL路径")
    parser.add_argument("--output", required=True, help="输出JSONL路径")
    parser.add_argument("--req-per-question", type=int, default=1, help="每问题生成答案数（默认1）")
    parser.add_argument("--clean-tmp", action="store_true", help="清理临时文件")
    args = parser.parse_args()

    if args.req_per_question < 1:
        log_message("错误：req-per-question必须≥1")
        sys.exit(1)

    try:
        if not os.path.exists(args.input_path):
            log_message(f"错误：输入文件不存在 {args.input_path}")
            sys.exit(1)
        
        # 创建输出目录
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 启动处理
        optimizer = APIOptimizer(req_per_question=args.req_per_question)
        await optimizer.process_jsonl(args.input_path, args.output, args.clean_tmp)

    except Exception as e:
        log_message(f"全局异常 | {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main_async())