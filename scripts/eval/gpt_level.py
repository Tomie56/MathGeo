import asyncio
import json
import os
import sys
import re
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

# ------------------------------ 配置参数 ------------------------------
API_KEY = "495e7f4ae82ddc5ccdb928b1bb686375"
BASE_URL = "https://dl.yunstorm.com/v1"
MODEL_NAME = "qwen3-vl-235b-a22b-instruct"  

MAX_WORKERS = 64
MAX_RETRY = 3

# ------------------------------ 核心提示词 ------------------------------
ENGLISH_DIFFICULTY_PROMPT = """
You are a geometry teacher with 10+ years of experience, evaluating geometric problem difficulty for academic assessment.

### Evaluation Rules
1. Evaluate ONLY the geometric problem provided after this prompt
2. Assess difficulty via TWO core dimensions:
   - Graph Complexity: geometric elements count, spatial relationships, labeling clarity
   - Problem Difficulty: knowledge threshold, reasoning steps, misleading conditions
3. Assign a difficulty score (integer 1-100): 1 = easiest (elementary level), 100 = hardest (competition level)

### Output Requirements (MANDATORY)
- Strictly wrap the integer in LaTeX \\boxed{} format (e.g., \\boxed{45})
- Return NOTHING except the boxed integer (no explanations, text, or symbols)
"""

# ------------------------------ 工具函数 ------------------------------
def log_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

def parse_boxed_difficulty(raw_result):
    if not raw_result:
        log_message("模型返回结果为空，无法解析难度值")
        return None
    
    pattern = r"\\?boxed\{(\d+)\}"
    match = re.search(pattern, raw_result.strip(), re.IGNORECASE)
    if not match:
        log_message(f"未检测到boxed格式，原始返回：{raw_result[:50]}...")
        return None
    
    try:
        difficulty = int(match.group(1))
        return max(1, min(100, difficulty))
    except ValueError:
        log_message(f"boxed内非有效数字：{match.group(1)}")
        return None

def calculate_judge_ratio(all_items):
    """
    统计所有数据（含跳过的）：
    1. 将1-100的gpt_level按「等数量」分10档（每档样本数≈总样本数/10），标注为diff_gpt_level（1-10）
    2. 按diff_gpt_level统计judge为true的比例
    """
    # 第一步：筛选有效样本（有gpt_level且为1-100的数字）
    valid_items = []
    for item in all_items:
        level = item.get("gpt_level")
        if isinstance(level, (int, float)) and 1 <= level <= 100:
            valid_items.append({
                "gpt_level": int(level),
                "judge": item.get("judge", False),
                "raw_item": item  # 保留原始项（可选）
            })
    
    if not valid_items:
        log_message("无有效gpt_level（1-100）样本，跳过统计")
        return {}
    
    # 第二步：按gpt_level排序，计算等数量分档的边界
    valid_items_sorted = sorted(valid_items, key=lambda x: x["gpt_level"])
    total_valid = len(valid_items_sorted)
    items_per_bin = math.ceil(total_valid / 10)  # 每档的样本数（向上取整保证分满10档）
    
    # 第三步：为每个样本分配diff_gpt_level（1-10，等数量分档）
    level_stats = defaultdict(lambda: {"total": 0, "true_count": 0})
    for idx, item in enumerate(valid_items_sorted):
        # 核心逻辑：按排序后的索引分配档级，确保每档样本数均衡
        diff_level = min(10, (idx // items_per_bin) + 1)
        level_stats[diff_level]["total"] += 1
        if item["judge"] is True:
            level_stats[diff_level]["true_count"] += 1
    
    # 第四步：输出统计结果（按diff_gpt_level升序）
    ratio_result = {}
    for diff_level in sorted(level_stats.keys()):
        total = level_stats[diff_level]["total"]
        true_cnt = level_stats[diff_level]["true_count"]
        ratio_result[diff_level] = true_cnt / total if total > 0 else 0.0
        # 补充打印每档的gpt_level范围，便于验证
        start_idx = (diff_level - 1) * items_per_bin
        end_idx = min(diff_level * items_per_bin, total_valid) - 1
        min_level = valid_items_sorted[start_idx]["gpt_level"]
        max_level = valid_items_sorted[end_idx]["gpt_level"]
        log_message(
            f"难度档{diff_level}：样本数{total} | judge=true数量{true_cnt} | 比例{ratio_result[diff_level]:.2%} | 原gpt_level范围[{min_level}, {max_level}]"
        )
    return ratio_result

# ------------------------------ 核心类 ------------------------------
class DifficultyEvaluator:
    def __init__(self):
        self.client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
        )
        self.request_queue = asyncio.Queue()
        self.start_time = datetime.now()
        self.success_count = 0  # 处理成功的样本数
        self.failure_count = 0  # 处理失败的样本数
        self.skip_count = 0     # 已有gpt_level跳过的样本数
        self.total_count = 0    # 总有效样本数（含跳过+处理）
        self.processed_count = 0# 实际处理的样本数（不含跳过）
        self.permission_error_count = 0
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self.processed_items = []  # 本次处理成功的样本
        self.all_items = []        # 所有有效样本（含跳过的）

    async def get_image_url(self, image_path):
        if image_path.startswith(("http://", "https://", "s3://")):
            return image_path
        try:
            mime_type, _ = guess_type(image_path)
            mime_type = mime_type or f"image/{image_path.split('.')[-1].lower()}"
            with open(image_path, "rb") as f:
                base64_data = base64.b64encode(f.read()).decode("utf-8")
            return f"data:{mime_type};base64,{base64_data}"
        except Exception as e:
            log_message(f"图片转base64失败：{image_path} | 错误{str(e)}")
            return None

    async def construct_messages(self, human_text, image_path):
        image_content = []
        img_url = await self.get_image_url(image_path)
        if img_url:
            image_content.append({
                "type": "image_url",
                "image_url": {"url": img_url}
            })
        
        full_prompt = f"""
        {ENGLISH_DIFFICULTY_PROMPT}

        Problem Text:
        {human_text}
        """
        text_content = {"type": "text", "text": full_prompt.strip()}
        return [{"role": "user", "content": image_content + [text_content]}]

    def call_api_sync(self, messages):
        """修复None下标访问错误"""
        try:
            completion = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                max_tokens=50,
                top_p=1.0,
                stop=None
            )
            # 关键修复：先判断completion是否为None
            if completion is None or not completion.choices:
                log_message("API返回空的choices列表")
                return None
            return completion.choices[0].message.content.strip()
        except Exception as e:
            error_msg = str(e).lower()
            if "permission" in error_msg or "auth" in error_msg:
                self.permission_error_count += 1
                log_message(f"权限错误：{error_msg} | 累计次数{self.permission_error_count}")
            else:
                log_message(f"API调用失败：{str(e)}")
            return None

    async def call_api_async(self, idx, human_text, image_path):
        messages = await self.construct_messages(human_text, image_path)
        if not messages:
            log_message(f"样本{idx}：消息构造失败")
            return None

        if self.permission_error_count >= 3:
            log_message(f"样本{idx}：权限错误超限，停止调用")
            return None

        for retry in range(MAX_RETRY):
            try:
                loop = asyncio.get_running_loop()
                raw_result = await loop.run_in_executor(
                    self.executor, self.call_api_sync, messages
                )
                difficulty = parse_boxed_difficulty(raw_result)
                if difficulty is not None:
                    return difficulty
                log_message(f"样本{idx}：第{retry+1}次返回格式无效，重试中...")
                await asyncio.sleep((retry + 1) ** 2)
            except Exception as e:
                log_message(f"样本{idx}：第{retry+1}次调用异常：{str(e)}")
                await asyncio.sleep((retry + 1) ** 2)
        return None

    async def worker(self, pbar):
        while True:
            task_retrieved = False
            try:
                idx, item = await self.request_queue.get()
                task_retrieved = True
                start_time = datetime.now()

                human_text = next(
                    (conv["value"] for conv in item.get("conversations", []) if conv["from"] == "human"),
                    None
                )
                image_path = item.get("image")

                if not human_text or not image_path:
                    log_message(f"样本{idx}：缺少human文本或image路径，跳过")
                    self.failure_count += 1
                    continue

                difficulty = await self.call_api_async(idx, human_text, image_path)
                if difficulty is not None:
                    item["gpt_level"] = difficulty
                    self.success_count += 1
                    self.processed_items.append(item)
                    status = "成功"
                else:
                    self.failure_count += 1
                    status = "失败"

                elapsed = (datetime.now() - start_time).total_seconds()
                self.processed_count += 1
                speed = self.processed_count / (datetime.now() - self.start_time).total_seconds()
                pbar.update(1)
                pbar.set_postfix({
                    "成功": self.success_count,
                    "失败": self.failure_count,
                    "跳过": self.skip_count,
                    "速度": f"{speed:.2f}条/s",
                    "权限错误": self.permission_error_count
                })
                log_message(f"样本{idx} | 状态{status} | 耗时{elapsed:.2f}s | 难度值{difficulty}")

            except asyncio.CancelledError:
                log_message(f"工作线程终止 | 样本{idx if task_retrieved else '未知'}")
                break
            except Exception as e:
                log_message(f"样本异常 | 业务错误：{str(e)}")
                self.failure_count += 1
                self.processed_count += 1
                pbar.update(1)
            finally:
                if task_retrieved:
                    self.request_queue.task_done()

    async def process_jsonl(self, input_path, output_path, max_lines):
        """核心流程：读取（仅前max_lines行+跳过已有gpt_level）→ 处理 → 合并保存 → 全量统计"""
        # 1. 读取输入JSONL（仅前max_lines行 + 跳过已有gpt_level的样本）
        log_message(f"读取输入文件（仅前{max_lines}行）：{input_path}")
        items_to_process = []  # 需要处理的样本
        self.all_items = []    # 所有有效样本（含跳过）
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # 仅处理前max_lines行
                if line_num > max_lines:
                    log_message(f"已读取前{max_lines}行，停止读取")
                    break
                    
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    self.all_items.append(item)
                    self.total_count += 1

                    # 关键逻辑：如果已有gpt_level且非空，跳过处理
                    if "gpt_level" in item and item["gpt_level"] is not None:
                        self.skip_count += 1
                        log_message(f"行{line_num}：已有gpt_level={item['gpt_level']}，跳过处理")
                        continue
                    
                    items_to_process.append(item)
                except json.JSONDecodeError:
                    log_message(f"行{line_num}：JSON解析失败，跳过")

        # 2. 校验样本数量
        if self.total_count == 0:
            log_message("无有效样本，退出")
            return
        log_message(f"读取完成 | 总读取行数：{min(line_num-1, max_lines)} | 有效样本数：{self.total_count} | 跳过样本数：{self.skip_count} | 待处理样本数：{len(items_to_process)}")
        
        if len(items_to_process) == 0:
            log_message("无待处理样本，直接统计并退出")
            calculate_judge_ratio(self.all_items)
            return

        # 3. 填充任务队列
        for idx, item in enumerate(items_to_process):
            await self.request_queue.put((idx, item))

        # 4. 启动并发处理
        with tqdm(total=len(items_to_process), desc="难度评估", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}") as pbar:
            workers = [asyncio.create_task(self.worker(pbar)) for _ in range(min(MAX_WORKERS, len(items_to_process)))]
            await self.request_queue.join()
            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

        # 5. 合并结果（原有样本 + 本次处理成功的样本）
        log_message(f"合并结果：原有样本({self.total_count}) + 本次处理成功({len(self.processed_items)})")
        # 构建样本ID映射（用image字段唯一标识，可根据实际调整）
        item_map = {item.get("image"): item for item in self.all_items}
        for processed_item in self.processed_items:
            img_key = processed_item.get("image")
            if img_key in item_map:
                item_map[img_key] = processed_item  # 覆盖原有样本的gpt_level

        # 6. 保存合并后的全量结果
        log_message(f"保存全量结果到{output_path}")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
            for item in self.all_items:
                img_key = item.get("image")
                final_item = item_map.get(img_key, item)
                await f.write(json.dumps(final_item, ensure_ascii=False) + "\n")

        # 7. 全量统计judge=true比例（含跳过的样本）
        log_message("\n=== 全量样本（含跳过）judge=true比例统计 ===")
        calculate_judge_ratio(self.all_items)

        # 8. 最终汇总
        log_message("\n=== 处理完成汇总 ===")
        log_message(f"总读取行数：{min(line_num-1, max_lines)}")
        log_message(f"有效样本数：{self.total_count}")
        log_message(f"跳过样本数：{self.skip_count}（已有gpt_level）")
        log_message(f"实际处理样本数：{self.processed_count}")
        log_message(f"处理成功数：{self.success_count} | 处理失败数：{self.failure_count}")
        log_message(f"处理成功率：{self.success_count/self.processed_count:.2%}" if self.processed_count > 0 else "处理成功率：0%")
        if self.permission_error_count > 0:
            log_message(f"⚠️  存在{self.permission_error_count}次权限错误，请检查API Key和模型权限")

async def main_async():
    parser = argparse.ArgumentParser(description="英文Prompt驱动的数学题难度评估工具（输出boxed格式难度值）")
    parser.add_argument("--input",
                        default="/mnt/afs/jingjinhao/project/GeoChain/MathGeo/A500_v4/output/all_test_answer_level_shuffled.jsonl",
                        help="输入JSONL文件路径")
    parser.add_argument("--output",
                        default="/mnt/afs/jingjinhao/project/GeoChain/MathGeo/A500_v4/output/all_test_answer_gpt_level_dis.jsonl",
                        help="输出带gpt_level的JSONL路径")
    # 新增max-lines参数，默认值1000
    parser.add_argument("--max-lines",
                        type=int,
                        default=2000,
                        help="仅处理输入文件的前N行，默认1000")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        log_message(f"错误：输入文件{args.input}不存在")
        sys.exit(1)

    evaluator = DifficultyEvaluator()
    # 传入max_lines参数
    await evaluator.process_jsonl(args.input, args.output, args.max_lines)

if __name__ == "__main__":
    asyncio.run(main_async())