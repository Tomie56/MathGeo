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
from openai import AsyncOpenAI, APIError, Timeout, APIConnectionError  # 导入OpenAI异步客户端及异常

def log_message(message):
    """增强型日志记录（与全流程脚本保持一致）"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

# 基础配置（改为API Key模式）
API_KEY = "sk-118c7a1f568f42ee93bc0cdf5b95fe17"  # API Key
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 接口基础地址
MODEL = "qwen3-vl-235b-a22b-thinking"  # 模型名称
MAX_CONCURRENT = 1  # 最大并发数
max_retry = 3
server_status = defaultdict(lambda: {"pending": 0, "response_time": 1.0})  # 保留状态跟踪

# 修正AOSS客户端配置路径（适配当前用户目录）
try:
    from aoss_client import client
    _aoss_client = client.Client('/mnt/afs/jingjinhao/aoss.conf')
except Exception as e:
    log_message(f"AOSS客户端初始化警告：{str(e)} | 若无需S3图片支持，可忽略此警告")
    _aoss_client = None

class APIOptimizer:
    def __init__(self, req_per_question):
        self.request_queue = asyncio.Queue()
        self.start_time = datetime.now()
        self.success_count = 0
        self.failure_count = 0
        self.total_processed = 0
        self.req_per_question = req_per_question  # 从参数接收生成答案数量
        # 初始化OpenAI异步客户端
        self.client = AsyncOpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
        )

    async def encode_image_to_base64(self, image_path):
        """优化图片编码逻辑：增加路径有效性检查"""
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
        """优化MIME类型判断，增加编码失败处理"""
        base64_data = await self.encode_image_to_base64(image_path)
        if not base64_data:
            return None
        
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            ext = image_path.split('.')[-1].lower()
            mime_map = {'webp': 'image/webp', 'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg'}
            mime_type = mime_map.get(ext, 'application/octet-stream')
        
        return f"data:{mime_type};base64,{base64_data}"

    async def dynamic_load_balancer(self):
        """简化负载均衡（单接口模式）"""
        return BASE_URL  # 直接返回基础地址

    async def construct_messages(self, image_paths, item):
        """优化提示词：明确几何问题求解要求，关联gt字段信息"""
        prompt = (
            "Solve the following geometric math problem step by step. "
            "First, analyze the geometric elements and relationships from the image and question. "
            "Show all formulas, calculations, and logical deductions clearly — do not skip any steps. "
            "Ensure your solution aligns with the geometric context provided. "
            "At the end of your response, place the final numerical answer inside a LaTeX box using \\boxed{}.\n\n"
            "Make sure that the final answer uses LaTeX-style expressions and is wrapped in \\boxed{}."
        )

        # 构建图片URL列表（过滤编码失败的图片）
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

    async def call_openai_api_async(self, idx, image_paths, item):
        """修改为OpenAI客户端调用方式"""
        messages = await self.construct_messages(image_paths, item)
        if not messages:
            log_message(f"样本{idx} | 无法构建有效请求（无图片或参数错误）")
            return None

        for retry in range(max_retry):
            try:
                # 记录当前请求状态
                server_status[BASE_URL]["pending"] += 1
                start_time = datetime.now()

                # 调用OpenAI异步客户端
                response = await self.client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=16384,
                    timeout=2000  # 超时设置（秒）
                )

                # 更新响应时间统计
                rt = (datetime.now() - start_time).total_seconds()
                server_status[BASE_URL]["response_time"] = 0.9 * server_status[BASE_URL]["response_time"] + 0.1 * rt

                return response.choices[0].message.content

            except (APIConnectionError, Timeout) as e:
                log_message(f"样本{idx} | 网络异常（重试{retry+1}/{max_retry}）| 类型：{type(e).__name__} | 错误：{str(e)}")
            except APIError as e:
                log_message(f"样本{idx} | API异常（重试{retry+1}/{max_retry}）| 状态码：{e.status_code} | 错误：{e.message}")
            except KeyError as e:
                log_message(f"样本{idx} | 响应格式错误（重试{retry+1}/{max_retry}）| 缺失字段：{str(e)}")
            except Exception as e:
                log_message(f"样本{idx} | 未处理异常（重试{retry+1}/{max_retry}）| 类型：{type(e).__name__} | 错误：{str(e)}")
            finally:
                server_status[BASE_URL]["pending"] -= 1
                await asyncio.sleep((1 + retry)**2)  # 退避策略
        return None

    def is_answer_valid(self, answer):
        """新增答案有效性校验：检查是否包含LaTeX框"""
        if not answer:
            return False
        return '\\boxed{' in answer

    async def worker(self, pbar, tmp_dir):
        """优化工作线程：移除aiohttp session（改用客户端内置连接）"""
        while True:
            task_retrieved = False
            try:
                idx, item = await self.request_queue.get()
                task_retrieved = True
                output_json = os.path.join(tmp_dir, f"{idx}.json")
                start_time = datetime.now()

                # 跳过已存在且有效结果的文件
                if os.path.exists(output_json):
                    try:
                        async with aiofiles.open(output_json, 'r') as f:
                            existing_data = json.loads(await f.read())
                            valid_answers = [ans for ans in existing_data.get("generated_answer", []) if self.is_answer_valid(ans)]
                            if len(valid_answers) >= self.req_per_question:
                                self.total_processed += 1
                                pbar.update(1)
                                self.request_queue.task_done()
                                continue
                    except Exception as e:
                        log_message(f"样本{idx} | 中间文件校验异常 | 错误：{str(e)} | 重新生成")

                # 处理图片路径（兼容单图/多图）
                image_paths = item['image'] if isinstance(item['image'], list) else [item['image']]
                image_paths = [os.path.abspath(path) for path in image_paths]

                # API调用生成答案（筛选有效结果）
                results = []
                retry_history = []
                for attempt in range(self.req_per_question + max_retry):
                    result = await self.call_openai_api_async(idx, image_paths, item)
                    if result:
                        if self.is_answer_valid(result):
                            results.append(result)
                            retry_history.append(f"第{attempt+1}次：成功（有效）")
                            if len(results) >= self.req_per_question:
                                break
                        else:
                            retry_history.append(f"第{attempt+1}次：成功, 无效，无boxed")
                    else:
                        retry_history.append(f"第{attempt+1}次：失败")

                # 写入中间文件（原子操作）
                tmp_path = f"{output_json}.tmp"
                async with aiofiles.open(tmp_path, 'w', encoding='utf-8') as f:
                    item['generated_answer'] = results[:self.req_per_question]
                    item['answer_validity'] = [self.is_answer_valid(ans) for ans in results[:self.req_per_question]]
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

                # 更新进度条
                pbar.set_postfix({
                    '成功': self.success_count,
                    '失败': self.failure_count,
                    '速度': f"{self.total_processed/(datetime.now()-self.start_time).total_seconds():.2f}条/s",
                    '负载': f"{server_status[BASE_URL]['pending']}",
                    '累计处理': self.total_processed
                })
                pbar.update(1)

            except asyncio.CancelledError:
                log_message(f"工作线程正常取消 | 样本{idx if task_retrieved else '未知'} | 已处理：{self.total_processed}条")
                break
            except Exception as e:
                log_message(f"样本{idx if task_retrieved else '未知'} | 工作线程异常 | 错误：{str(e)}")
                traceback.print_exc()
            finally:
                if task_retrieved:
                    self.request_queue.task_done()

    async def process_jsonl(self, input_path, output_path, clean_tmp):
        """优化数据处理流程：增强输入校验，支持临时文件清理"""
        # 1. 读取并校验输入数据
        log_message(f"读取输入文件：{input_path}")
        items = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # 增强字段校验：确保gt.expr存在（呼应format.py的筛选）
                    required_fields = ['image', 'generated_question', 'gt']
                    if all(field in data for field in required_fields):
                        if len(data['generated_question']) == 0:
                            log_message(f"跳过行{line_num}：generated_question为空")
                            continue
                        if 'expr' not in data['gt']:
                            log_message(f"跳过行{line_num}：gt字段缺少expr")
                            continue
                        items.append(data)
                    else:
                        log_message(f"跳过行{line_num}：缺少核心字段（{', '.join(set(required_fields)-set(data.keys()))}）")
                except json.JSONDecodeError:
                    log_message(f"跳过行{line_num}：JSON解析失败")
                except Exception as e:
                    log_message(f"跳过行{line_num}：处理异常 | 错误：{str(e)}")

        total_items = len(items)
        log_message(f"输入文件读取完成 | 有效条目：{total_items} | 需生成答案数/条目：{self.req_per_question}")
        if total_items == 0:
            log_message("无有效条目可处理，退出")
            return

        # 2. 初始化临时目录
        tmp_dir = os.path.join(os.path.dirname(output_path), "tmp_answer")
        os.makedirs(tmp_dir, exist_ok=True)
        log_message(f"中间文件目录：{tmp_dir}")

        # 3. 填充任务队列
        for idx, item in enumerate(items):
            await self.request_queue.put((idx, item))

        # 4. 启动工作线程处理任务（无需aiohttp session，使用客户端内置连接）
        with tqdm(total=total_items, desc="生成答案", file=sys.stdout) as pbar:
            worker_count = min(MAX_CONCURRENT, total_items)
            workers = [asyncio.create_task(self.worker(pbar, tmp_dir)) for _ in range(worker_count)]
            
            await self.request_queue.join()
            # 取消所有工作线程
            for w in workers:
                if not w.done():
                    w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

        # 5. 汇总结果到输出文件
        log_message(f"汇总结果到：{output_path}")
        success_count = 0
        async with aiofiles.open(output_path, 'w', encoding='utf-8') as outf:
            for idx in range(total_items):
                json_file = os.path.join(tmp_dir, f"{idx}.json")
                if os.path.exists(json_file):
                    try:
                        async with aiofiles.open(json_file, 'r') as f:
                            item = json.loads(await f.read())
                            if len(item.get("generated_answer", [])) >= self.req_per_question:
                                await outf.write(json.dumps(item, ensure_ascii=False) + '\n')
                                success_count += 1
                    except Exception as e:
                        log_message(f"跳过异常中间文件（序号{idx}）| 错误：{str(e)}")

        # 6. 清理临时文件（如果启用）
        if clean_tmp:
            import shutil
            try:
                shutil.rmtree(tmp_dir)
                log_message(f"已清理临时目录：{tmp_dir}")
            except Exception as e:
                log_message(f"清理临时目录失败 | 错误：{str(e)}")

        # 7. 输出最终统计
        log_message(f"===== 处理完成 =====")
        log_message(f"总条目：{total_items} | 成功生成：{success_count} | 成功率：{success_count/total_items:.1%}")
        log_message(f"最终文件路径：{output_path}")

async def main_async():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="调用API生成数学问题答案（增强版）")
    parser.add_argument("input_path", help="输入JSONL路径（get_question.py输出）")
    parser.add_argument("--output", required=True, help="输出JSONL路径")
    parser.add_argument("--req-per-question", type=int, default=1, help="每个问题生成的答案数量（默认1）")
    parser.add_argument("--clean-tmp", action="store_true", help="处理完成后清理临时文件")
    args = parser.parse_args()

    # 校验参数有效性
    if args.req_per_question < 1:
        log_message("错误：--req-per-question必须大于等于1")
        sys.exit(1)

    try:
        # 验证输入文件存在
        if not os.path.exists(args.input_path):
            log_message(f"错误：输入文件不存在 -> {args.input_path}")
            sys.exit(1)
        
        # 创建输出目录
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            log_message(f"创建输出目录：{output_dir}")

        # 启动处理流程
        optimizer = APIOptimizer(req_per_question=args.req_per_question)
        await optimizer.process_jsonl(args.input_path, args.output, args.clean_tmp)

    except Exception as e:
        log_message(f"全局异常 | 类型：{type(e).__name__} | 错误：{str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main_async())