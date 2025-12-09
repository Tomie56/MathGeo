import json
import math
import sys
from collections import defaultdict
from datetime import datetime

# ------------------------------ 工具函数 ------------------------------
def log_message(message):
    """带时间戳的日志打印"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

def calculate_balanced_judge_ratio(file_path, output_file_path):
    """
    从JSONL文件读取数据，按gpt_level等数量分10档：
    1. 统计每档judge=true的比例
    2. 为每个样本添加gpt_level_dis字段（分档后的标签1-10）
    3. 保存包含新标签的JSONL文件
    :param file_path: 输入JSONL文件路径
    :param output_file_path: 输出包含gpt_level_dis的JSONL文件路径
    :return: 分档统计结果字典
    """
    # 第一步：读取所有样本（保留原始数据）+ 筛选有效样本
    all_items = []          # 所有原始样本（含无效/无gpt_level的）
    valid_items = []        # 有有效gpt_level的样本（带行号）
    total_lines = 0
    invalid_lines = 0
    no_gpt_level = 0

    log_message(f"开始读取文件：{file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            raw_line = line.strip()
            if not raw_line:
                invalid_lines += 1
                all_items.append(None)  # 标记空行
                continue
            
            try:
                item = json.loads(raw_line)
                all_items.append(item)  # 保留原始项
                # 筛选有gpt_level且为1-100整数的样本
                gpt_level = item.get("gpt_level")
                if not (isinstance(gpt_level, (int, float)) and 1 <= gpt_level <= 100):
                    no_gpt_level += 1
                    continue
                
                valid_items.append({
                    "gpt_level": int(gpt_level),
                    "judge": item.get("judge", False),
                    "line_num": line_num - 1,  # 对应all_items的索引（从0开始）
                    "raw_item": item
                })
            except json.JSONDecodeError:
                log_message(f"行{line_num}：JSON解析失败，跳过")
                invalid_lines += 1
                all_items.append(None)  # 标记解析失败行

    # 打印读取统计
    log_message(f"读取完成 | 总行数：{total_lines} | 无效行（空/解析失败）：{invalid_lines} | 无有效gpt_level：{no_gpt_level} | 有效样本数：{len(valid_items)}")
    
    if not valid_items:
        log_message("无有效样本，退出统计")
        return {}

    # 第二步：按gpt_level排序，等数量分10档
    valid_items_sorted = sorted(valid_items, key=lambda x: x["gpt_level"])
    total_valid = len(valid_items_sorted)
    items_per_bin = math.ceil(total_valid / 10)  # 每档样本数（向上取整）
    
    # 第三步：1.统计每档比例 2.为有效样本分配gpt_level_dis
    level_stats = defaultdict(lambda: {
        "total": 0,
        "true_count": 0,
        "min_gpt_level": None,
        "max_gpt_level": None,
        "line_nums": []  # 记录该档样本的行号
    })
    # 存储有效样本的分档结果（行索引 → gpt_level_dis）
    dis_label_map = {}

    for idx, item in enumerate(valid_items_sorted):
        diff_level = min(10, (idx // items_per_bin) + 1)
        # 更新分档统计
        level_stats[diff_level]["total"] += 1
        if item["judge"] is True:
            level_stats[diff_level]["true_count"] += 1
        if level_stats[diff_level]["min_gpt_level"] is None or item["gpt_level"] < level_stats[diff_level]["min_gpt_level"]:
            level_stats[diff_level]["min_gpt_level"] = item["gpt_level"]
        if level_stats[diff_level]["max_gpt_level"] is None or item["gpt_level"] > level_stats[diff_level]["max_gpt_level"]:
            level_stats[diff_level]["max_gpt_level"] = item["gpt_level"]
        level_stats[diff_level]["line_nums"].append(item["line_num"] + 1)  # 恢复为原始行号（从1开始）
        
        # 记录分档标签（行索引 → diff_level）
        dis_label_map[item["line_num"]] = diff_level

    # 第四步：为所有样本添加gpt_level_dis字段（无效样本设为None）
    log_message(f"为样本添加gpt_level_dis字段...")
    output_items = []
    for idx, item in enumerate(all_items):
        if item is None:
            output_items.append(None)
            continue
        # 给有效样本赋值分档标签，无效样本设为None
        item["gpt_level_dis"] = dis_label_map.get(idx, None)
        output_items.append(item)

    # 第五步：保存包含gpt_level_dis的JSONL文件
    log_message(f"保存包含gpt_level_dis的文件到：{output_file_path}")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for item in output_items:
            if item is None:
                f.write("\n")  # 保留空行
                continue
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 第六步：输出统计结果
    log_message("\n=== 分档统计结果（等数量分10档）===")
    ratio_result = {}
    for diff_level in sorted(level_stats.keys()):
        stats = level_stats[diff_level]
        total = stats["total"]
        true_cnt = stats["true_count"]
        ratio = true_cnt / total if total > 0 else 0.0
        ratio_result[diff_level] = ratio
        
        log_message(
            f"难度档{diff_level}（gpt_level_dis={diff_level}）："
            f"  - 样本数：{total} | judge=true数量：{true_cnt} | 比例：{ratio:.2%}"
            # f"  - 原gpt_level范围：[{stats['min_gpt_level']}, {stats['max_gpt_level']}]\n"
            # f"  - 样本行号示例：{stats['line_nums'][:5]}..." if len(stats['line_nums']) > 5 else f"  - 样本行号：{stats['line_nums']}"
        )

    # 总统计
    total_true = sum([stats["true_count"] for stats in level_stats.values()])
    overall_ratio = total_true / total_valid if total_valid > 0 else 0.0
    log_message(f"\n=== 整体统计 ===")
    log_message(f"有效样本总数：{total_valid} | judge=true总数：{total_true} | 整体比例：{overall_ratio:.2%}")
    log_message(f"\n包含gpt_level_dis的文件已保存：{output_file_path}")

    return ratio_result

# ------------------------------ 主函数 ------------------------------
if __name__ == "__main__":
    # 输入输出文件路径配置
    INPUT_FILE = "/mnt/afs/jingjinhao/project/GeoChain/MathGeo/A500_v4/output/all_test_answer_gpt_level_dis.jsonl"
    # 输出文件在原路径基础上加"_with_dis_label"后缀，避免覆盖原文件
    OUTPUT_FILE = "/mnt/afs/jingjinhao/project/GeoChain/MathGeo/A500_v4/output/all_test_answer_gpt_level_dis_with_dis_label.jsonl"
    
    # 执行统计+添加分档标签
    result = calculate_balanced_judge_ratio(INPUT_FILE, OUTPUT_FILE)
    
    # 可选：将统计结果保存为JSON文件
    stat_output = "judge_ratio_statistics_with_dis_label.json"
    with open(stat_output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log_message(f"统计结果已保存到：{stat_output}")