import json
import random
import argparse
import os
from datetime import datetime

def log_message(message):
    """打印带时间戳的日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")

def shuffle_jsonl(input_path, output_path, seed=42):
    """
    打乱JSONL文件的行顺序（整体读取，无分块）
    :param input_path: 输入JSONL文件路径
    :param output_path: 输出打乱后的JSONL文件路径
    :param seed: 随机种子（保证结果可复现）
    """
    # 1. 校验输入文件
    if not os.path.exists(input_path):
        log_message(f"错误：输入文件{input_path}不存在")
        return False
    
    # 2. 初始化随机种子（保证可复现）
    random.seed(seed)
    log_message(f"设置随机种子：{seed}")

    # 3. 整体读取所有行（无分块）
    log_message(f"开始读取输入文件：{input_path}")
    all_lines = []
    valid_json_lines = 0
    invalid_lines = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            stripped_line = line.strip()
            # 保留空行（保证格式一致）
            if not stripped_line:
                all_lines.append(line)
                continue
            # 校验JSON格式
            try:
                json.loads(stripped_line)
                all_lines.append(line)
                valid_json_lines += 1
            except json.JSONDecodeError:
                log_message(f"行{line_num}：JSON格式无效，跳过")
                invalid_lines += 1
                continue
    
    log_message(f"读取完成 | 总行数（含空行）：{len(all_lines)} | 有效JSON行：{valid_json_lines} | 无效行：{invalid_lines}")

    # 4. 整体打乱行顺序
    log_message("开始打乱行顺序...")
    random.shuffle(all_lines)

    # 5. 写入输出文件
    log_message(f"开始写入输出文件：{output_path}")
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in all_lines:
            f.write(line)
        # 强制刷盘，确保数据写入磁盘
        f.flush()
        os.fsync(f.fileno())
    
    log_message(f"打乱完成！输出文件：{output_path}")
    return True

def main():
    # 配置命令行参数（无required约束）
    parser = argparse.ArgumentParser(description="打乱JSONL文件的行顺序（整体读取）")
    parser.add_argument("--input", 
                        default="/mnt/afs/jingjinhao/project/GeoChain/MathGeo/A500_v4/output/all_test_answer_level.jsonl",
                        help="输入JSONL文件路径（默认指定自定义路径）")
    parser.add_argument("--output", 
                        default="/mnt/afs/jingjinhao/project/GeoChain/MathGeo/A500_v4/output/all_test_answer_level_shuffled.jsonl",
                        help="输出打乱后的JSONL文件路径（默认指定自定义路径）")
    parser.add_argument("--seed", 
                        type=int, 
                        default=42,
                        help="随机种子（默认42，保证结果可复现）")
    
    args = parser.parse_args()

    # 执行打乱
    success = shuffle_jsonl(
        input_path=args.input,
        output_path=args.output,
        seed=args.seed
    )
    
    # 退出码：0成功，1失败
    exit(0 if success else 1)

if __name__ == "__main__":
    main()