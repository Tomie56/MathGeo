"""
启动脚本:
python ./scripts/call_api/convert.py <输入jsonl路径> --output <最终输出路径>

功能：仅处理expr字段中含:的真值（SymPy符号化转换），latex字段保持原样 + 原等价性判断
"""

import json
import os
import sys
from datetime import datetime
import argparse
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import sympy  # SymPy符号化计算核心
from aoss_client import client

# 导入参考方法中的核心函数
sys.path.append('/mnt/afs/liangjinwei/project/verl/verl/utils/reward_score/omni_reward/math')
from utils import extract_answer, grade_answer_mathd, grade_answer_sympy

def log_message(message):
    """日志输出函数，带时间戳"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

# -------------------------- 核心：仅处理含:的expr字段（无LaTeX处理） --------------------------
def calculate_ratio_sympy(val):
    """
    仅针对含:的字符串执行SymPy符号化比例转换（无LaTeX处理）：
    1. 清理空格 → 2. 分割:前后 → 3. SymPy解析计算 → 4. 简化结果
    无:/非字符串/计算异常 → 返回原字符串
    """
    # 仅处理字符串且含:的场景
    if not isinstance(val, str) or ':' not in val:
        return val
    
    try:
        # 步骤1：清理所有空格
        val_clean = val.replace(" ", "")
        
        # 步骤2：按第一个:分割为前后两部分
        ratio_parts = val_clean.split(':', 1)
        if len(ratio_parts) != 2 or not ratio_parts[0] or not ratio_parts[1]:
            log_message(f"⚠️  expr比例分割无效（单边为空），返回原值：{val[:50]}")
            return val
        
        left_part, right_part = ratio_parts[0], ratio_parts[1]
        
        # 步骤3：SymPy符号化解析
        left_expr = sympy.sympify(left_part)
        right_expr = sympy.sympify(right_part)
        
        # 步骤4：除零防护
        if sympy.simplify(right_expr) == 0:
            log_message(f"⚠️  expr比例分母为0，返回原值：{val[:50]}")
            return val
        
        # 步骤5：计算比值并简化
        ratio_result = sympy.simplify(left_expr / right_expr)
        
        # 转换为字符串返回
        return str(ratio_result)
    
    except sympy.SympifyError as e:
        log_message(f"⚠️  expr SymPy解析失败[{str(e)[:30]}]，返回原值：{val[:50]}")
        return val
    except ZeroDivisionError:
        log_message(f"⚠️  expr比例计算除零错误，返回原值：{val[:50]}")
        return val
    except Exception as e:
        log_message(f"⚠️  expr比例计算异常[{str(e)[:30]}]，返回原值：{val[:50]}")
        return val

# -------------------------- 处理Ground Truth（仅处理expr，latex保持原样） --------------------------
def get_ground_truths(gt_dict):
    """
    仅处理expr字段（含:则转换），latex字段直接保留原值，返回真值列表
    """
    ground_truths = []
    if not isinstance(gt_dict, dict):
        return ground_truths
    
    # 仅处理expr字段（核心：更新原字典的expr值）
    expr_original = gt_dict.get("expr", "").strip()
    if expr_original:
        expr_converted = calculate_ratio_sympy(expr_original)
        gt_dict["expr"] = expr_converted  # 强制更新expr字段
        ground_truths.append(expr_converted)
        if expr_converted != expr_original:
            log_message(f"✅ expr更新：{expr_original[:50]} → {expr_converted[:50]}")
    
    # latex字段完全保持原样，不做任何转换
    latex_original = gt_dict.get("latex", "").strip()
    if latex_original:
        ground_truths.append(latex_original)
    
    return ground_truths

# -------------------------- 初始化AOSS客户端 --------------------------
try:
    _aoss_client = client.Client('/mnt/afs/jingjinhao/aoss.conf')
except Exception as e:
    log_message(f"⚠️  AOSS客户端初始化警告：{str(e)} | S3图片宽高无法获取")
    _aoss_client = None

# -------------------------- 图片尺寸获取（保留原有逻辑） --------------------------
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
        log_message(f"❌ 无效图片路径：{str(image_path)[:100]}")
        return (None, None)
    except Exception as e:
        log_message(f"❌ 获取图片尺寸失败：{str(e)}")
        return (None, None)

# -------------------------- 核心处理函数（保留原有格式+比对逻辑） --------------------------
def process_item(item):
    """处理单条数据，保持原有输出格式"""
    # 1. 提取图片信息
    image = item.get("image", "")
    image_path = []
    if isinstance(image, list):
        image_path = [img.strip() for img in image if img.strip()]
    elif isinstance(image, str) and image.strip():
        image_path = [image.strip()]
    width, height = get_image_size(image_path[0] if image_path else None)
    
    # 2. 提取对话内容
    generated_question = item.get("generated_question", [])
    question = generated_question[0].strip() if (isinstance(generated_question, list) and generated_question) else ""
    
    generated_answer = item.get("generated_answer", [])
    answer_text = generated_answer[0].strip() if (isinstance(generated_answer, list) and generated_answer) else ""
    conversations = [
        {"from": "human", "value": question},
        {"from": "gpt", "value": answer_text}
    ]
    
    # 3. 提取答案
    extracted_answer = extract_answer(answer_text)
    log_message(f"📌 提取答案：{extracted_answer if extracted_answer else 'None'}")
    
    # 4. 处理真值（仅更新expr，latex原样）+ 等价性判断
    gt = item.get("gt", {})
    ground_truths = get_ground_truths(gt)  # gt["expr"]已更新，latex原样加入列表
    judge = False
    
    if extracted_answer and ground_truths:
        for truth in ground_truths:
            # 沿用原有等价性判断逻辑
            is_correct = grade_answer_mathd(extracted_answer, truth) or grade_answer_sympy(extracted_answer, truth)
            if is_correct:
                judge = True
                break
        log_message(f"🔍 等价性判断：{'正确' if judge else '错误'}（真值列表：{ground_truths}）")
    
    # 5. 返回结果（含更新后的gt）
    return {
        "image": image_path[0] if image_path else "",
        "width": width,
        "height": height,
        "conversations": conversations,
        "answer": extracted_answer if extracted_answer else "", 
        "gt": gt,  # 仅expr更新，latex保持原样
        "judge": judge,
        "diff": item.get("diff", "unknown")
    }

# -------------------------- 主函数（保留原有流程） --------------------------
def main():
    parser = argparse.ArgumentParser(description="仅处理expr含:的真值，latex保持原样")
    parser.add_argument("input_path", help="输入JSONL路径")
    parser.add_argument("--output", required=True, help="输出JSONL路径")
    args = parser.parse_args()

    # 输入文件检查
    if not os.path.exists(args.input_path):
        log_message(f"❌ 错误：输入文件不存在 → {args.input_path}")
        sys.exit(1)

    # 输出目录创建
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 数据处理
    log_message(f"🚀 开始处理：{args.input_path}")
    total_lines = 0
    valid_lines = 0
    correct_count = 0

    with open(args.input_path, 'r', encoding='utf-8') as fin:
        lines = [line.strip() for line in fin if line.strip()]
        total_lines = len(lines)
        log_message(f"📊 共读取 {total_lines} 条数据")

    with open(args.output, 'w', encoding='utf-8') as fout:
        for line_num, line in enumerate(tqdm(lines, desc="处理进度"), 1):
            try:
                item = json.loads(line)
                # 必要字段检查
                required_fields = ["generated_question", "generated_answer", "image", "gt"]
                if not all(field in item for field in required_fields):
                    missing = [f for f in required_fields if f not in item]
                    log_message(f"⚠️  行{line_num}：缺少字段 {missing}，填充默认值")
                
                processed = process_item(item)
                fout.write(json.dumps(processed, ensure_ascii=False) + "\n")
                valid_lines += 1
                if processed["judge"]:
                    correct_count += 1

            except json.JSONDecodeError:
                log_message(f"❌ 行{line_num}：JSON解析失败，生成默认结构")
                default_data = {
                    "image": "",
                    "width": None,
                    "height": None,
                    "conversations": [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}],
                    "answer": "",
                    "gt": {},
                    "judge": False,
                    "diff": "unknown",
                }
                fout.write(json.dumps(default_data, ensure_ascii=False) + "\n")
            except Exception as e:
                log_message(f"❌ 行{line_num}：处理失败 → {str(e)}，生成默认结构")
                default_data = {
                    "image": "",
                    "width": None,
                    "height": None,
                    "conversations": [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}],
                    "answer": "",
                    "gt": {},
                    "judge": False,
                    "diff": "unknown",
                }
                fout.write(json.dumps(default_data, ensure_ascii=False) + "\n")

    # 统计输出
    log_message("\n===== 处理完成 =====")
    log_message(f"📈 总数据：{total_lines} | 有效处理：{valid_lines} | 错误：{total_lines - valid_lines}")
    if valid_lines > 0:
        accuracy = (correct_count / valid_lines) * 100
        log_message(f"🎯 答案正确率：{correct_count}/{valid_lines} ({accuracy:.2f}%)")
    else:
        log_message(f"📉 无有效数据可计算正确率")
    log_message(f"💾 输出路径：{args.output}")

if __name__ == "__main__":
    # 依赖安装：pip install pillow tqdm sympy
    main()