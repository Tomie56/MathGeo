import os
import json
import argparse
from datetime import datetime
import sys
import numpy as np

def log_message(message):
    """日志记录函数"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

def collect_diff_values(input_path):
    """收集所有有效diff值（用于计算分位数）"""
    diff_values = []
    total_entries = 0
    valid_diff_entries = 0
    error_entries = 0

    log_message(f"开始收集diff值：{input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            total_entries += 1
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "diff" not in data:
                    log_message(f"行 {line_num}：缺失'diff'字段，跳过")
                    continue
                diff_val = data["diff"]
                if not isinstance(diff_val, (int, float)):
                    log_message(f"行 {line_num}：'diff'不是数值（{type(diff_val)}），跳过")
                    continue
                diff_values.append(diff_val)
                valid_diff_entries += 1
            except json.JSONDecodeError:
                error_entries += 1
                log_message(f"行 {line_num}：JSON解析错误，跳过")
            except Exception as e:
                error_entries += 1
                log_message(f"行 {line_num}：处理错误 - {str(e)}，跳过")

    log_message(f"diff收集完成：总条目={total_entries}，有效diff={valid_diff_entries}，错误={error_entries}")
    if valid_diff_entries == 0:
        log_message("错误：无有效diff值，无法生成level")
        return None
    return diff_values

def calculate_quantile_thresholds(diff_values):
    """计算分位数阈值（确保最高diff对应level 10）"""
    sorted_diffs = sorted(diff_values)  # 升序排序（从小到大）
    n = len(sorted_diffs)
    log_message(f"有效diff数量：{n}，计算分位数阈值（10等级划分）")

    # 计算9个阈值（对应10%-90%分位数），作为1-9等级的上限
    thresholds = []
    for i in range(1, 10):
        quantile = i / 10.0  # 10%, 20%, ..., 90%
        threshold = np.percentile(sorted_diffs, quantile * 100, interpolation='higher')  # 确保阈值不低于分位实际值
        thresholds.append(threshold)

    # 打印阈值（明确各等级边界）
    log_message("等级划分阈值（小于等于阈值属于对应等级，超过则进入更高等级）：")
    for i, thresh in enumerate(thresholds, 1):
        log_message(f"level {i} 上限（{i*10}%分位数）：{thresh:.4f}")
    log_message(f"level 10：所有 > {thresholds[-1]:.4f} 的diff值（最高的10%）")
    return thresholds

def discretize_to_level(diff_val, thresholds):
    """将diff值映射到1-10（10为最高）"""
    for level, thresh in enumerate(thresholds, 1):
        if diff_val <= thresh:
            return level
    return 10  # 最高等级

def process_file(input_path, output_path):
    """主处理流程：收集diff→算阈值→映射level→统计judge比例→保存"""
    # 1. 收集有效diff值
    diff_values = collect_diff_values(input_path)
    if diff_values is None:
        return False

    # 2. 计算分位数阈值
    thresholds = calculate_quantile_thresholds(diff_values)

    # 3. 初始化level统计器（记录每个level的judge情况）
    # 结构：{level: {'total': 总条目数, 'true_count': judge为true的数量}}
    level_stats = {i: {'total': 0, 'true_count': 0} for i in range(1, 11)}

    # 4. 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        log_message(f"创建输出目录：{output_dir}")

    # 5. 遍历文件生成level并统计judge
    total_entries = 0
    success_entries = 0  # 成功生成level的条目
    valid_judge_entries = 0  # 有有效judge字段的条目（用于整体统计）
    error_entries = 0

    log_message("开始生成level并统计judge...")
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            total_entries += 1
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # 检查diff字段
                if "diff" not in data:
                    log_message(f"行 {line_num}：无'diff'，跳过")
                    continue
                diff_val = data["diff"]
                if not isinstance(diff_val, (int, float)):
                    log_message(f"行 {line_num}：'diff'非数值，跳过")
                    continue

                # 计算level
                level = discretize_to_level(diff_val, thresholds)
                data["level"] = level
                success_entries += 1

                # 统计judge为true的比例（仅处理有效judge字段）
                judge_val = data.get("judge")
                if isinstance(judge_val, bool):
                    # 更新该level的统计
                    level_stats[level]['total'] += 1
                    if judge_val:
                        level_stats[level]['true_count'] += 1
                    valid_judge_entries += 1
                else:
                    log_message(f"行 {line_num}：'judge'无效（非布尔值），不计入统计")

                # 写入输出
                outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

                # 进度提示
                if success_entries % 1000 == 0:
                    log_message(f"已处理 {success_entries} 条，当前行 {line_num}")

            except json.JSONDecodeError:
                error_entries += 1
                log_message(f"行 {line_num}：JSON解析错误，跳过")
            except Exception as e:
                error_entries += 1
                log_message(f"行 {line_num}：处理错误 - {str(e)}，跳过")

    # 6. 计算并输出judge统计结果
    log_message("\n===== judge为true的比例统计 =====")
    log_message(f"有效统计样本（含有效judge字段）：{valid_judge_entries}")
    # 按level排序输出
    for level in range(1, 11):
        stats = level_stats[level]
        total = stats['total']
        true_count = stats['true_count']
        ratio = (true_count / total) * 100 if total > 0 else 0
        log_message(
            f"level {level}："
            f"总样本数={total}，"
            f"judge为true={true_count}，"
            f"比例={ratio:.1f}%"
        )

    # 7. 输出整体处理统计
    log_message("\n===== 数据处理完成 =====")
    log_message(f"总条目：{total_entries} | 成功生成level：{success_entries} | 错误/跳过：{error_entries}")
    log_message(f"输出文件：{output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="生成level并统计不同level中judge为true的比例（10为最高等级）")
    parser.add_argument("--input", 
                       default="/mnt/afs/jingjinhao/project/GeoChain/MathGeo/A500_v4_instruct/output/A500_v4_instruct.jsonl",
                       help="输入JSONL路径（默认提供的路径）")
    parser.add_argument("--output", 
                       default="/mnt/afs/jingjinhao/project/GeoChain/MathGeo/A500_v4_instruct/output/A500_v4_instruct_level.jsonl",
                       help="输出路径（默认原目录生成带_judge_stats的文件）")
    args = parser.parse_args()

    # 处理输出路径
    if not args.output:
        input_dir = os.path.dirname(args.input)
        input_filename = os.path.basename(args.input)
        name, ext = os.path.splitext(input_filename)
        args.output = os.path.join(input_dir, f"{name}_judge_stats{ext}")

    # 检查输入文件
    if not os.path.exists(args.input):
        log_message(f"错误：输入文件不存在 - {args.input}")
        sys.exit(1)

    # 检查numpy依赖
    try:
        import numpy
    except ImportError:
        log_message("错误：需安装numpy（运行 'pip install numpy'）")
        sys.exit(1)

    log_message(f"输入文件：{args.input}")
    log_message(f"输出文件：{args.output}")
    process_file(args.input, args.output)

if __name__ == "__main__":
    main()