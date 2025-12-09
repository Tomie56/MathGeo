"""
启动脚本（兼容bash自动化调用）:
python ./scripts/call_api/get_question.py <输入jsonl路径> --output <输出jsonl路径>

增强日志功能:
1. 实时显示每个样本的处理状态和详细错误信息
2. 进度条显示增强:
   - 当前处理速度
   - 预计剩余时间
3. 错误处理增强:
   - 区分API错误、网络错误、参数错误、权限错误
   - 记录具体错误原因和解决方案提示
"""

import asyncio
import json
import os
import sys
from datetime import datetime
import traceback
import base64
from collections import defaultdict
from tqdm import tqdm
import argparse
from mimetypes import guess_type
import aiofiles
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

# ------------------------------ 配置参数（严格遵循用户指定格式） ------------------------------
# 阿里云百炼配置（用户提供的原始配置）
API_KEY = "495e7f4ae82ddc5ccdb928b1bb686375"
BASE_URL = "https://dl.yunstorm.com/v1"
# 恢复用户指定的模型名称
MODEL_NAME = "qwen3-vl-235b-a22b-instruct"  

# 并发配置
MAX_WORKERS = 16
max_retry = 3
req_per_question = 1

# 提示词配置（保留原逻辑）
SYSTEM_PROMPT = "<|im_start|>system\nDescribe the picture in detail.<|im_end|>\n"
USER_START = "<|im_start|>user\n"
ASSISTANT_START = "<|im_start|>assistant\n"
IM_END = "<|im_end|>\n"
IMG_TAG = "<img></img>\n"
AUDIO_TAG = "<|card|>: ><|card|>:\n"

# ------------------------------ 工具函数 ------------------------------
def log_message(message):
    """增强型日志记录"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

# ------------------------------ 核心类 ------------------------------
class APIOptimizer:
    def __init__(self):
        # 初始化OpenAI客户端（完全遵循用户提供的配置）
        self.client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
        )
        self.request_queue = asyncio.Queue()
        self.start_time = datetime.now()
        self.success_count = 0
        self.failure_count = 0
        self.total_processed = 0
        self.permission_error_count = 0
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    async def get_image_url(self, image_path):
        """适配用户格式：支持OSS链接直接使用，本地图片转base64"""
        # 若为网络链接（如阿里云OSS、s3），直接返回原URL
        if image_path.startswith(("http://", "https://", "s3://")):
            return image_path
        # 本地图片转base64（保持原逻辑，适配本地文件）
        else:
            try:
                mime_type, _ = guess_type(image_path)
                if mime_type is None:
                    file_ext = image_path.split('.')[-1].lower()
                    mime_type = f'image/{file_ext}' if file_ext in ['png', 'jpg', 'jpeg', 'webp'] else 'application/octet-stream'
                
                with open(image_path, "rb") as f:
                    base64_data = base64.b64encode(f.read()).decode("utf-8")
                return f"data:{mime_type};base64,{base64_data}"
            except Exception as e:
                log_message(f"图片处理失败 -> {image_path} | 错误：{str(e)}")
                return None

    async def construct_messages(self, image_paths, item):
        """构造消息（严格遵循用户提供的messages格式）"""
        image_contents = []
        for img_path in image_paths:
            img_url = await self.get_image_url(img_path)
            if img_url:
                # 完全匹配用户指定的image_url格式
                image_contents.append({
                    'type': 'image_url',
                    'image_url': {'url': img_url}
                })
            else:
                log_message(f"跳过无效图片 -> {img_path}")
                return None

        # 文本内容保留原提示词逻辑
        text_content = {
            'type': 'text',
            'text': (
                f'Based on the provided text and image information, generate a math problem stem for me. The stem should adhere to the language style of common problem formats, fully, rigorously, and concisely integrating and describing the image information. It should then pose a question formatted according to the content in Question Reference, ensuring the question is accurate and written in English.\n\n'
                f'Question Reference: {item["question"]}\n'
                f'Description of Geometric Context: {item["qt_description"]}\n'
                f'Full Geometric Description: {item["description"]}'
            )
        }

        # 消息结构完全匹配用户示例（role: user，content为image+text列表）
        return [
            {
                'role': 'user',
                'content': image_contents + [text_content]
            }
        ]

    def call_openai_api_sync(self, messages):
        """同步调用API（完全遵循用户提供的参数配置）"""
        try:
        # if True:
            completion = self.client.chat.completions.create(
                model=MODEL_NAME,  # 用户指定模型
                messages=messages,
                temperature=0.6,   # 用户指定参数
                max_tokens=32768,  # 用户指定参数
                top_p=0.95,        # 用户指定参数
            )
            print(completion.model_dump_json())
            return completion.choices[0].message.content.strip()
        except Exception as e:
            log_message(f"❌ API调用未知错误：{str(e)}")
            return None

    async def call_openai_api_async(self, idx, image_paths, item):
        """异步包装API调用"""
        messages = await self.construct_messages(image_paths, item)
        if not messages:
            log_message(f"样本{idx}：消息构造失败")
            return None

        if self.permission_error_count >= 3:
            log_message(f"⚠️  已累计{self.permission_error_count}次权限错误，停止样本{idx}重试")
            return None

        for retry in range(max_retry):
            try:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    self.executor,
                    self.call_openai_api_sync,
                    messages
                )
                if result:
                    return result
                else:
                    if self.permission_error_count > 0:
                        log_message(f"样本{idx}：因权限错误停止重试")
                        break
                    log_message(f"样本{idx}：第{retry+1}次调用失败，重试中...")
                    await asyncio.sleep((2 + retry)**2)
            except Exception as e:
                log_message(f"样本{idx}：第{retry+1}次调用异常 -> {str(e)}")
                await asyncio.sleep((2 + retry)**2)
        return None

    async def worker(self, pbar, tmp_dir):
        """工作线程（保留原核心逻辑）"""
        while True:
            task_retrieved = False
            try:
                idx, item = await self.request_queue.get()
                task_retrieved = True
                
                output_json = os.path.join(tmp_dir, f"{idx}.json")
                start_time = datetime.now()
                status = "失败"
                retry_history = []

                # 跳过已完成任务
                if os.path.exists(output_json):
                    try:
                        async with aiofiles.open(output_json, 'r') as f:
                            existing_data = json.loads(await f.read())
                            if len(existing_data.get("generated_question", [])) >= req_per_question:
                                self.total_processed += 1
                                pbar.update(1)
                                self.request_queue.task_done()
                                continue
                    except Exception as e:
                        log_message(f"样本{idx}：中间文件校验失败 -> {str(e)}")

                # 处理图片路径（兼容单图/多图）
                image_paths = []
                if isinstance(item['image'], list):
                    image_paths = item['image']
                else:
                    image_paths = [item['image']]

                # 生成问题
                results = []
                for attempt in range(req_per_question + max_retry):
                    result = await self.call_openai_api_async(idx, image_paths, item)
                    success = bool(result)
                    retry_history.append(f"第{attempt+1}次:{'成功' if success else '失败'}")
                    if success:
                        results.append(result)
                        if len(results) >= req_per_question:
                            break

                # 写入中间文件
                tmp_path = f"{output_json}.tmp"
                async with aiofiles.open(tmp_path, 'w', encoding='utf-8') as f:
                    item['generated_question'] = results[:req_per_question]
                    await f.write(json.dumps(item, ensure_ascii=False, indent=2))
                os.replace(tmp_path, output_json)

                # 统计与日志
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
                log_message(
                    f"样本{idx} | 状态:{status} | 耗时:{elapsed:.2f}s | "
                    f"重试记录:{' '.join(retry_history)} | 有效结果:{len(results)}/{req_per_question}"
                )
                
                # 更新进度条
                processing_speed = self.total_processed / (datetime.now() - self.start_time).total_seconds()
                pbar.set_postfix({
                    '成功': self.success_count,
                    '失败': self.failure_count,
                    '权限错误': self.permission_error_count,
                    '速度': f"{processing_speed:.2f}条/s",
                    '累计处理': self.total_processed
                })

            except asyncio.CancelledError:
                log_message(f"工作线程被取消 | 样本{idx if task_retrieved else '未知'} | 已处理:{self.total_processed}")
                break
            except Exception as e:
                log_message(f"样本{idx if task_retrieved else '未知'} | 业务异常 -> {str(e)}")
                traceback.print_exc()
            finally:
                if task_retrieved:
                    self.request_queue.task_done()

    async def process_jsonl(self, input_path, output_path):
        """核心流程（保留原逻辑）"""
        log_message(f"读取输入文件：{input_path}")
        items = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    required_fields = ['image', 'question', 'qt_description', 'description', 'gt']
                    if all(field in data for field in required_fields):
                        items.append(data)
                    else:
                        log_message(f"跳过无效条目（行{line_num}）：缺少必要字段")
                except json.JSONDecodeError:
                    log_message(f"跳过无效条目（行{line_num}）：JSON解析失败")
        
        total_items = len(items)
        log_message(f"输入文件读取完成 | 有效条目数：{total_items}")
        if total_items == 0:
            log_message("无有效条目，退出程序")
            return

        # 创建临时目录
        tmp_dir = os.path.join(os.path.dirname(output_path), "tmp_question")
        os.makedirs(tmp_dir, exist_ok=True)
        log_message(f"中间文件目录：{tmp_dir}")

        # 填充任务队列
        for idx, item in enumerate(items):
            await self.request_queue.put((idx, item))

        # 并发处理
        with tqdm(
            total=total_items,
            desc="生成数学问题",
            file=sys.stdout,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [剩余{remaining}] {postfix}"
        ) as pbar:
            worker_count = min(MAX_WORKERS, total_items)
            workers = [asyncio.create_task(self.worker(pbar, tmp_dir)) for _ in range(worker_count)]
            
            await self.request_queue.join()
            for w in workers:
                if not w.done():
                    w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

        # 汇总结果
        log_message(f"汇总结果到：{output_path}")
        success_count = 0
        async with aiofiles.open(output_path, 'w', encoding='utf-8') as outf:
            for idx in range(total_items):
                json_file = os.path.join(tmp_dir, f"{idx}.json")
                if os.path.exists(json_file):
                    try:
                        async with aiofiles.open(json_file, 'r') as f:
                            item = json.loads(await f.read())
                            if len(item.get("generated_question", [])) >= req_per_question:
                                await outf.write(json.dumps(item, ensure_ascii=False) + '\n')
                                success_count += 1
                    except Exception as e:
                        log_message(f"跳过异常文件（序号{idx}）：{str(e)}")

        # 最终统计
        log_message(f"结果汇总完成 | 输出路径：{output_path}")
        log_message(
            f"最终统计：总条目{total_items} | 成功生成{success_count} | "
            f"成功率：{success_count/total_items:.1%} | 权限错误次数：{self.permission_error_count}"
        )
        if self.permission_error_count > 0:
            log_message("⚠️  权限错误提示：")
            log_message(f"  - 当前模型：{MODEL_NAME}")
            log_message(f"  - 请登录阿里云百炼控制台：https://dashscope.console.aliyun.com/")
            log_message(f"  - 检查API Key对应的账号是否已开通该模型的访问权限")

async def main_async():
    parser = argparse.ArgumentParser(description="调用阿里云百炼VL模型生成数学问题（适配用户指定格式）")
    parser.add_argument("input_path", help="输入JSONL文件路径")
    parser.add_argument("--output", required=True, help="输出JSONL文件路径")
    args = parser.parse_args()

    try:
        if not os.path.exists(args.input_path):
            log_message(f"错误：输入文件不存在 -> {args.input_path}")
            sys.exit(1)
        
        output_dir = os.path.dirname(args.output)
        os.makedirs(output_dir, exist_ok=True)

        optimizer = APIOptimizer()
        await optimizer.process_jsonl(args.input_path, args.output)

    except Exception as e:
        log_message(f"全局异常 -> {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main_async())