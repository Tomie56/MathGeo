import os
import json
import shutil
import argparse
from datetime import datetime
import sys
from collections import defaultdict

def log_message(message):
    """与原脚本保持一致的日志格式"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

def is_answer_valid(answer):
    """验证答案是否包含LaTeX框"""
    return isinstance(answer, str) and '\\boxed{' in answer

def extract_valid_answers(item, req_per_question):
    """提取符合要求的有效答案"""
    generated_answers = item.get('generated_answer', [])
    # 优先使用原有效性标记，无标记时重新校验
    if 'answer_validity' in item:
        valid_mask = item['answer_validity']
        valid_answers = [ans for ans, valid in zip(generated_answers, valid_mask) if valid]
    else:
        valid_answers = [ans for ans in generated_answers if is_answer_valid(ans)]
    return valid_answers[:req_per_question]

def main():
    parser = argparse.ArgumentParser(description="提取并整理中间答案（从临时文件）")
    parser.add_argument("--tmp-dir", help="临时文件目录路径（默认：输出文件同级的tmp_answer）")
    parser.add_argument("--output", required=True, help="整理后的输出JSONL路径")
    parser.add_argument("--req-per-question", type=int, default=1, help="每个问题需要的有效答案数量（默认1）")
    parser.add_argument("--clean-tmp", action="store_true", help="整理完成后清理临时文件")
    args = parser.parse_args()

    # 验证参数有效性
    if args.req_per_question < 1:
        log_message("错误：--req-per-question必须大于等于1")
        sys.exit(1)

    # 确定临时目录路径
    if not args.tmp_dir:
        output_dir = os.path.dirname(args.output)
        args.tmp_dir = os.path.join(output_dir, "tmp_answer")
    log_message(f"使用临时目录：{args.tmp_dir}")

    # 检查临时目录是否存在
    if not os.path.exists(args.tmp_dir):
        log_message(f"错误：临时目录不存在 -> {args.tmp_dir}")
        sys.exit(1)

    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        log_message(f"创建输出目录：{output_dir}")

    # 收集并筛选临时文件（仅处理数字命名的json文件）
    tmp_files = []
    file_errors = defaultdict(list)
    for filename in os.listdir(args.tmp_dir):
        if filename.endswith(".json"):
            try:
                # 提取原始行号（文件名格式：<original_line_num>.json）
                line_num = int(filename[:-5])
                file_path = os.path.join(args.tmp_dir, filename)
                tmp_files.append((line_num, file_path))
            except ValueError:
                file_errors["无效文件名"].append(filename)

    # 按原始行号排序
    tmp_files.sort(key=lambda x: x[0])
    total_files = len(tmp_files)
    log_message(f"发现临时文件：{total_files}个（有效命名）")
    for err_type, files in file_errors.items():
        log_message(f"跳过{err_type}：{len(files)}个文件（例：{files[0]}）")

    # 处理临时文件并写入输出
    success_count = 0
    with open(args.output, 'w', encoding='utf-8') as outf:
        for line_num, file_path in tmp_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    item = json.load(f)
                
                # 提取有效答案
                valid_answers = extract_valid_answers(item, args.req_per_question)
                if len(valid_answers) >= args.req_per_question:
                    # 更新item中的答案为筛选后的有效结果
                    item['generated_answer'] = valid_answers
                    item['answer_validity'] = [is_answer_valid(ans) for ans in valid_answers]
                    outf.write(json.dumps(item, ensure_ascii=False) + '\n')
                    success_count += 1
                else:
                    log_message(f"行号{line_num}：有效答案不足（{len(valid_answers)}/{args.req_per_question}），跳过")
            
            except json.JSONDecodeError:
                log_message(f"行号{line_num}：JSON解析失败，跳过")
            except Exception as e:
                log_message(f"行号{line_num}：处理异常 | {str(e)}，跳过")

    # 清理临时文件
    if args.clean_tmp:
        try:
            shutil.rmtree(args.tmp_dir)
            log_message(f"已清理临时目录：{args.tmp_dir}")
        except Exception as e:
            log_message(f"清理临时目录失败 | {str(e)}")

    # 输出统计结果
    log_message("===== 整理完成 =====")
    log_message(f"总临时文件：{total_files} | 成功整理：{success_count} | 整理率：{success_count/total_files:.1%}" if total_files > 0 else "无有效临时文件")
    log_message(f"输出文件路径：{args.output}")

if __name__ == "__main__":
    main()