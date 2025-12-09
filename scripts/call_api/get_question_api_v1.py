import asyncio
import aiofiles
import json
import os
import itertools
from collections import defaultdict
import sys
from datetime import datetime
import traceback
import base64
from tqdm import tqdm
from mimetypes import guess_type
import argparse
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

# 基础配置（更新为instruct模型）
max_retry = 3
DEFAULT_API_KEY = "495e7f4ae82ddc5ccdb928b1bb686375"
DEFAULT_BASE_URL = "https://dl.yunstorm.com/v1"
DEFAULT_MODEL = "qwen3-vl-235b-a22b-instruct"  # 替换为instruct模型

# 尝试初始化AOSS客户端（兼容S3图片）
try:
    from aoss_client import client as aoss_client_cls
    _aoss_client = aoss_client_cls.Client('/mnt/afs/jingjinhao/aoss.conf')
except ImportError:
    _aoss_client = None
except Exception as e:
    print(f"AOSS客户端初始化警告：{str(e)}")
    _aoss_client = None

def log_message(message):
    """增强型日志记录"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

class APIOptimizer:
    def __init__(self, req_per_question, threads, model, api_key, base_url):
        self.request_queue = asyncio.Queue()
        self.start_time = datetime.now()
        self.success_count = 0
        self.failure_count = 0
        self.total_processed = 0
        self.permission_error_count = 0  # 新增：权限错误统计
        self.req_per_question = req_per_question
        self.threads = threads
        self.model = model
        self.loop = None  # 延迟绑定事件循环
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=500
        )
        # 自定义线程池（兼容低版本Python）
        self.executor = ThreadPoolExecutor(max_workers=self.threads)

    async def encode_image_to_base64(self, image_path):
        """图片转Base64（兼容S3/本地路径）"""
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
            log_message(f"图片编码失败：{image_path} | 错误：{str(e)}")
            return None

    async def local_image_to_data_url(self, image_path):
        """图片转Data URL（带MIME类型）"""
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
        """构建API请求的messages（适配instruct模型）"""
        prompt = (
            "Solve the following geometric math problem based on the provided image and question.\n"
            "At the end of your response, place the final numerical answer inside a LaTeX box using \\boxed{}.\n\n"
            "Make sure that the final answer uses LaTeX-style expressions and is wrapped in \\boxed{}."
        )

        # 构建图片URL列表
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
            log_message("无有效图片可用于生成答案")
            return None

        # 适配instruct模型的message格式
        messages = [
            {
                'role': 'user',
                'content': image_urls + [
                    {
                        'type': 'text',
                        'text': prompt + f'Question: {item["generated_question"][0]}'
                    }
                ]
            }
        ]
        return messages

    def is_answer_valid(self, answer):
        """答案有效性校验（仅校验answer）"""
        if not answer:
            return False
        return '\\boxed{' in answer

    def call_openai_sync(self, messages):
        """同步调用OpenAI API（适配instruct模型，移除thinking）"""
        full_content = ""
        try:
            # 非流式调用（instruct模型更适配）
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=8192,
                top_p=0.95,
                stream=False  # 关闭流式，直接返回完整结果
            )
            full_content = completion.choices[0].message.content.strip()
            return {
                "answer": full_content  # 仅返回answer，移除thinking
            } if full_content else None

        except Exception as e:
            # 细分错误类型：权限/API/网络/参数
            error_type = type(e).__name__
            error_msg = str(e)
            if "permission" in error_msg.lower() or "403" in error_msg:
                self.permission_error_count += 1
                log_message(f"权限错误 | 模型：{self.model} | 错误：{error_msg}")
            elif "network" in error_type.lower() or "timeout" in error_msg.lower():
                log_message(f"网络错误 | 错误：{error_msg}")
            elif "invalid" in error_msg.lower() or "parameter" in error_msg.lower():
                log_message(f"参数错误 | 错误：{error_msg}")
            else:
                log_message(f"API调用异常 | 类型：{error_type} | 错误：{error_msg}")
            return None

    async def call_api_with_retry(self, idx, image_paths, item):
        """带重试的API调用（适配instruct模型+权限错误处理）"""
        # 绑定事件循环（首次调用时初始化）
        if self.loop is None:
            self.loop = asyncio.get_running_loop()
        
        messages = await self.construct_messages(image_paths, item)
        if not messages:
            log_message(f"样本{idx} | 无法构建有效请求")
            return None

        # 权限错误累计≥3时停止重试
        if self.permission_error_count >= 3:
            log_message(f"⚠️  样本{idx} | 已累计{self.permission_error_count}次权限错误，停止重试")
            return None

        for retry in range(max_retry):
            try:
                # 核心改点：用loop.run_in_executor替代asyncio.to_thread（兼容Python3.9以下）
                result = await self.loop.run_in_executor(
                    self.executor,  # 自定义线程池
                    self.call_openai_sync,  # 同步函数
                    messages  # 函数参数
                )
                if result and self.is_answer_valid(result["answer"]):
                    return result
                else:
                    # 权限错误直接终止重试
                    if self.permission_error_count > 0:
                        log_message(f"样本{idx} | 第{retry+1}次重试 | 因权限错误停止")
                        break
                    log_message(f"样本{idx} | 重试{retry+1}/{max_retry} | 结果为空/无效")
                await asyncio.sleep((2 + retry) ** 2)  # 指数退避重试
            except Exception as e:
                log_message(f"样本{idx} | 第{retry+1}次调用异常 -> {str(e)}")
                await asyncio.sleep((2 + retry) ** 2)
        return None

    async def worker(self, pbar, tmp_dir):
        """工作线程（移除所有thinking相关逻辑）"""
        while True:
            task_retrieved = False
            try:
                idx, item = await self.request_queue.get()
                task_retrieved = True
                output_json = os.path.join(tmp_dir, f"{idx}.json")
                start_time = datetime.now()

                # 跳过已完成的有效文件（仅校验answer）
                if os.path.exists(output_json):
                    try:
                        async with aiofiles.open(output_json, 'r') as f:
                            existing_data = json.loads(await f.read())
                            valid_count = sum(1 for a in existing_data.get("generated_answer", []) if self.is_answer_valid(a))
                            if valid_count >= self.req_per_question:
                                self.total_processed += 1
                                pbar.update(1)
                                self.request_queue.task_done()
                                continue
                    except Exception as e:
                        log_message(f"样本{idx} | 中间文件校验异常 | {str(e)}")

                # 处理图片路径
                image_paths = item['image'] if isinstance(item['image'], list) else [item['image']]
                image_paths = [os.path.abspath(path) for path in image_paths]

                # 调用API生成结果（仅保留answer）
                results = []
                retry_history = []
                for attempt in range(self.req_per_question + max_retry):
                    result = await self.call_api_with_retry(idx, image_paths, item)
                    if result:
                        if self.is_answer_valid(result["answer"]):
                            results.append(result["answer"])  # 仅存answer
                            retry_history.append(f"第{attempt+1}次：成功（有效）")
                            if len(results) >= self.req_per_question:
                                break
                        else:
                            retry_history.append(f"第{attempt+1}次：成功（无boxed）")
                    else:
                        retry_history.append(f"第{attempt+1}次：失败")

                # 仅保留answer相关字段，移除thinking
                generated_answer = results[:self.req_per_question]
                answer_validity = [self.is_answer_valid(a) for a in generated_answer]

                # 写入临时文件（移除generated_thinking）
                tmp_path = f"{output_json}.tmp"
                async with aiofiles.open(tmp_path, 'w', encoding='utf-8') as f:
                    item['generated_answer'] = generated_answer
                    item['answer_validity'] = answer_validity
                    await f.write(json.dumps(item, ensure_ascii=False, indent=2))
                os.replace(tmp_path, output_json)

                # 统计状态
                elapsed = (datetime.now() - start_time).total_seconds()
                if len(results) >= self.req_per_question:
                    status = "成功"
                    self.success_count += 1
                elif len(results) > 0:
                    status = "部分成功"
                else:
                    status = "失败"
                    self.failure_count += 1

                self.total_processed += 1
                log_message(f"样本{idx} | 状态：{status} | 耗时：{elapsed:.2f}s | 重试记录：{' '.join(retry_history)} | 有效结果：{len(results)}/{self.req_per_question}")
                # 进度条新增权限错误统计
                pbar.set_postfix({
                    '成功': self.success_count,
                    '失败': self.failure_count,
                    '权限错误': self.permission_error_count,
                    '速度': f"{self.total_processed/(datetime.now()-self.start_time).total_seconds():.2f}条/s",
                    '线程数': self.threads,
                    '累计处理': self.total_processed
                })
                pbar.update(1)

            except asyncio.CancelledError:
                log_message(f"工作线程正常取消 | 样本{idx if task_retrieved else '未知'}")
                break
            except Exception as e:
                log_message(f"样本{idx if task_retrieved else '未知'} | 工作线程异常 | {str(e)}")
                traceback.print_exc()
            finally:
                if task_retrieved:
                    self.request_queue.task_done()

    async def process_jsonl(self, input_path, output_path, clean_tmp, start_line):
        """处理JSONL文件（移除thinking相关校验）"""
        log_message(f"读取输入文件：{input_path} | 从第{start_line}行开始处理 | 线程数：{self.threads} | 模型：{self.model}")
        valid_items = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    required_fields = ['image', 'generated_question', 'gt']
                    if all(field in data for field in required_fields):
                        if len(data['generated_question']) == 0:
                            log_message(f"跳过行{line_num}：generated_question为空")
                            continue
                        if 'expr' not in data['gt']:
                            log_message(f"跳过行{line_num}：gt字段缺少expr")
                            continue
                        valid_items.append((line_num, data))
                    else:
                        log_message(f"跳过行{line_num}：缺少核心字段")
                except json.JSONDecodeError:
                    log_message(f"跳过行{line_num}：JSON解析失败")
                except Exception as e:
                    log_message(f"跳过行{line_num}：处理异常 | {str(e)}")

        filtered_items = [item for item in valid_items if item[0] >= start_line]
        total_valid = len(valid_items)
        total_items = len(filtered_items)
        log_message(f"输入文件读取完成 | 总有效条目：{total_valid} | 待处理条目：{total_items}")
        if total_items == 0:
            log_message("无符合条件的条目可处理，退出")
            return

        # 初始化临时目录
        tmp_dir = os.path.join(os.path.dirname(output_path), "tmp_answer")
        os.makedirs(tmp_dir, exist_ok=True)
        log_message(f"中间文件目录：{tmp_dir}")

        # 填充任务队列
        for original_line_num, item in filtered_items:
            await self.request_queue.put((original_line_num, item))

        # 启动工作线程
        with tqdm(total=total_items, desc="生成答案（instruct模型）", file=sys.stdout) as pbar:
            workers = [asyncio.create_task(self.worker(pbar, tmp_dir)) for _ in range(self.threads)]
            await self.request_queue.join()
            # 取消工作线程
            for w in workers:
                if not w.done():
                    w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

        # 汇总结果（仅校验answer）
        log_message(f"汇总结果到：{output_path}")
        success_count = 0
        async with aiofiles.open(output_path, 'w', encoding='utf-8') as outf:
            for original_line_num, _ in filtered_items:
                json_file = os.path.join(tmp_dir, f"{original_line_num}.json")
                if os.path.exists(json_file):
                    try:
                        async with aiofiles.open(json_file, 'r') as f:
                            item = json.loads(await f.read())
                            if len(item.get("generated_answer", [])) >= self.req_per_question:
                                await outf.write(json.dumps(item, ensure_ascii=False) + '\n')
                                success_count += 1
                    except Exception as e:
                        log_message(f"跳过异常中间文件（行号{original_line_num}）| {str(e)}")

        # 清理临时文件
        if clean_tmp:
            import shutil
            try:
                shutil.rmtree(tmp_dir)
                log_message(f"已清理临时目录：{tmp_dir}")
            except Exception as e:
                log_message(f"清理临时目录失败 | {str(e)}")

        # 关闭线程池（新增：释放资源）
        self.executor.shutdown(wait=True)

        # 输出统计（新增权限错误提示）
        log_message(f"===== 处理完成 =====")
        log_message(f"总有效条目：{total_valid} | 处理条目：{total_items} | 成功生成：{success_count} | 成功率：{success_count/total_items:.1%}" if total_items > 0 else "无处理条目")
        log_message(f"最终文件路径：{output_path} | 线程数：{self.threads} | 模型：{self.model}")
        if self.permission_error_count > 0:
            log_message(f"⚠️  累计{self.permission_error_count}次权限错误，请检查API Key和模型访问权限！")

async def main_async():
    """主函数（无修改，仅适配模型默认值）"""
    parser = argparse.ArgumentParser(description="兼容版调用OpenAI生成数学问题答案（instruct模型）")
    parser.add_argument("input_path", help="输入JSONL路径")
    parser.add_argument("--output", required=True, help="输出JSONL路径")
    parser.add_argument("--req-per-question", type=int, default=1, help="每个问题生成的答案数量（默认1）")
    parser.add_argument("--clean-tmp", action="store_true", help="处理完成后清理临时文件")
    parser.add_argument("--start-line", type=int, default=1, help="从指定行号开始处理（默认1）")
    parser.add_argument("--threads", type=int, default=64, help="并发数（默认10，建议≤50）")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="模型名称")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY, help="API密钥")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL, help="API基础地址")
    args = parser.parse_args()

    # 参数校验
    if args.req_per_question < 1:
        log_message("错误：--req-per-question必须≥1")
        sys.exit(1)
    if args.start_line < 1:
        log_message("错误：--start-line必须≥1")
        sys.exit(1)
    if args.threads < 1 or args.threads > 200:
        log_message("错误：--threads必须在1-200之间")
        sys.exit(1)

    try:
        if not os.path.exists(args.input_path):
            log_message(f"错误：输入文件不存在 -> {args.input_path}")
            sys.exit(1)
        
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            log_message(f"创建输出目录：{output_dir}")

        optimizer = APIOptimizer(
            req_per_question=args.req_per_question,
            threads=args.threads,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url
        )
        await optimizer.process_jsonl(args.input_path, args.output, args.clean_tmp, args.start_line)

    except Exception as e:
        log_message(f"全局异常 | {type(e).__name__} | {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main_async())