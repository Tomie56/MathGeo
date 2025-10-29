"""
启动脚本:
python ./scripts/call_api/convert.py <输入jsonl路径> --output <最终输出路径>

功能：保持原有输出格式 + 参考方法提取答案 + 等价性判断
"""

import json
import re
import os
import sys
from datetime import datetime
import argparse
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from aoss_client import client

# 导入参考方法中的核心函数（确保路径正确）
sys.path.append('/mnt/afs/liangjinwei/project/verl/verl/utils/reward_score/omni_reward/math')
from utils import extract_answer, grade_answer_mathd, grade_answer_sympy

def log_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

# 初始化AOSS客户端
try:
    _aoss_client = client.Client('/mnt/afs/jingjinhao/aoss.conf')
except Exception as e:
    log_message(f"AOSS客户端警告：{str(e)} | S3图片宽高无法获取")
    _aoss_client = None

# -------------------------- 图片尺寸获取（保留原有功能） --------------------------
def get_image_size(image_path):
    if not image_path:
        return (None, None)
    try:
        if isinstance(image_path, str) and 's3' in image_path and _aoss_client:
            img_data = _aoss_client.get(image_path)
            with Image.open(BytesIO(img_data)) as img:
                return img.size
        if isinstance(image_path, str) and os.path.exists(image_path):
            with Image.open(image_path) as img:
                return img.size
        if isinstance(image_path, list):
            for path in image_path:
                if os.path.exists(path):
                    with Image.open(path) as img:
                        return img.size
                if 's3' in path and _aoss_client:
                    try:
                        img_data = _aoss_client.get(path)
                        with Image.open(BytesIO(img_data)) as img:
                            return img.size
                    except Exception:
                        continue
        log_message(f"无效图片路径：{str(image_path)[:100]}")
        return (None, None)
    except Exception as e:
        log_message(f"获取图片尺寸失败：{str(e)}")
        return (None, None)

# -------------------------- 处理Ground Truth（适配参考方法） --------------------------
def get_ground_truths(gt_dict):
    """从gt字典中提取有效真值列表（expr和latex）"""
    ground_truths = []
    if not isinstance(gt_dict, dict):
        return ground_truths
    # 优先取expr和latex字段作为真值
    for key in ["expr", "latex"]:
        val = gt_dict.get(key, "")
        if val and isinstance(val, str):
            ground_truths.append(val.strip())
    return ground_truths

# -------------------------- 核心处理函数（保持原有输出格式） --------------------------
def process_item(item):
    """处理单条数据，严格保持目标输出格式"""
    # 1. 提取图片信息（保留原有逻辑）
    image = item.get("image", "")
    image_path = []
    if isinstance(image, list):
        image_path = [img.strip() for img in image if img.strip()]
    elif isinstance(image, str) and image.strip():
        image_path = [image.strip()]
    width, height = get_image_size(image_path[0] if image_path else None)
    
    # 2. 提取对话内容（保持human/gpt格式）
    generated_question = item.get("generated_question", [])
    question = generated_question[0].strip() if (isinstance(generated_question, list) and generated_question) else ""
    
    generated_answer = item.get("generated_answer", [])
    answer_text = generated_answer[0].strip() if (isinstance(generated_answer, list) and generated_answer) else ""
    conversations = [
        {"from": "human", "value": question},
        {"from": "gpt", "value": answer_text}
    ]
    
    # 3. 提取答案（使用参考方法的extract_answer，自动处理boxed）
    extracted_answer = extract_answer(answer_text)
    log_message(f"提取答案：{extracted_answer[:50] if extracted_answer else 'None'}")
    
    # 4. 处理GT和等价性判断（使用参考方法的评分函数）
    gt = item.get("gt", {})
    ground_truths = get_ground_truths(gt)
    judge = False
    
    if extracted_answer and ground_truths:
        # 与每个真值比较，任一匹配则判定为正确
        for truth in ground_truths:
            is_correct = grade_answer_mathd(extracted_answer, truth) or grade_answer_sympy(extracted_answer, truth)
            if is_correct:
                judge = True
                break
        log_message(f"等价性判断：{'正确' if judge else '错误'}（真值：{ground_truths[:2]}）")
    
    # 5. 保持原有输出格式，字段完全对齐
    return {
        "image": image_path[0] if image_path else "",
        "width": width,
        "height": height,
        "conversations": conversations,
        "answer": extracted_answer if extracted_answer else "", 
        "gt": gt, 
        "judge": judge,
        "level": item.get("level", "unknown")
    }

# -------------------------- 主函数（保持原有流程） --------------------------
def main():
    parser = argparse.ArgumentParser(description="保持格式+参考方法判断答案等价性")
    parser.add_argument("input_path", help="输入JSONL路径")
    parser.add_argument("--output", required=True, help="最终输出JSONL路径")
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        log_message(f"错误：输入文件不存在 -> {args.input_path}")
        sys.exit(1)

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    log_message(f"开始处理：{args.input_path}")
    total_lines = 0
    valid_lines = 0
    correct_count = 0

    with open(args.input_path, 'r', encoding='utf-8') as fin:
        lines = [line.strip() for line in fin if line.strip()]
        total_lines = len(lines)
        log_message(f"共读取 {total_lines} 条数据")

    with open(args.output, 'w', encoding='utf-8') as fout:
        for line_num, line in enumerate(tqdm(lines, desc="转换并判断"), 1):
            try:
                item = json.loads(line)
                # 必要字段检查（确保输出格式完整）
                required_fields = ["generated_question", "generated_answer", "image", "gt"]
                if not all(field in item for field in required_fields):
                    missing = [f for f in required_fields if f not in item]
                    log_message(f"行{line_num}：缺少字段 {missing}，填充默认值")
                
                processed = process_item(item)
                fout.write(json.dumps(processed, ensure_ascii=False) + "\n")
                valid_lines += 1
                if processed["judge"]:
                    correct_count += 1

            except json.JSONDecodeError:
                log_message(f"行{line_num}：JSON解析失败，生成默认结构")
                # 解析失败时生成默认格式数据
                default_data = {
                    "image": "",
                    "width": None,
                    "height": None,
                    "conversations": [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}],
                    "answer": "",
                    "gt": {},
                    "judge": False,
                    "level": "unknown",
                }
                fout.write(json.dumps(default_data, ensure_ascii=False) + "\n")
            except Exception as e:
                log_message(f"行{line_num}：处理失败 -> {str(e)}，生成默认结构")
                default_data = {
                    "image": "",
                    "width": None,
                    "height": None,
                    "conversations": [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}],
                    "answer": "",
                    "gt": {},
                    "judge": False,
                    "level": "unknown",
                }
                fout.write(json.dumps(default_data, ensure_ascii=False) + "\n")

    # 输出统计结果
    log_message("\n===== 处理完成 =====")
    log_message(f"总数据：{total_lines} | 有效处理：{valid_lines} | 错误：{total_lines - valid_lines}")
    if valid_lines > 0:
        accuracy = (correct_count / valid_lines) * 100
        log_message(f"答案正确率：{correct_count}/{valid_lines} ({accuracy:.2f}%)")
    else:
        log_message("无有效数据可计算正确率")
    log_message(f"输出路径：{args.output}")

if __name__ == "__main__":
    # 需安装依赖：pip install pillow tqdm
    main()