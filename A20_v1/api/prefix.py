import json
import os
import sys
import argparse
from tqdm import tqdm
from datetime import datetime


def log(message):
    """带时间戳的日志输出"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def replace_image_path(image_str, new_str, old_str="results" ):
    """替换image路径中的指定字符串"""
    if not isinstance(image_str, str):
        return image_str  # 非字符串格式不处理
    return image_str.replace(old_str, new_str)


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="替换JSONL文件中image字段的'results'字符串")
    parser.add_argument("--input", 
                      default="/mnt/afs/jingjinhao/project/GeoChain/MathGeo/results/json/qa/qa_shaded_with_gt_20251102_122021_612.jsonl",
                      help="输入JSONL文件路径（默认提供目标文件路径）")
    parser.add_argument("--output", 
                        default="/mnt/afs/jingjinhao/project/GeoChain/MathGeo/results/json/qa/qa_shaded_with_gt_20251102_122021_612_modified.jsonl", help="输出JSONL文件路径")
    parser.add_argument("--replace-with", default="A500_v2",help="用于替换'results'的字符串")
    args = parser.parse_args()

    # 验证输入文件
    if not os.path.exists(args.input):
        log(f"错误：输入文件不存在 -> {args.input}")
        sys.exit(1)

    # 创建输出目录（如果需要）
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        log(f"创建输出目录 -> {output_dir}")

    # 统计信息
    total_lines = 0
    success_lines = 0
    error_lines = []

    # 读取并处理文件
    log(f"开始处理文件：{args.input}")
    with open(args.input, "r", encoding="utf-8") as fin:
        lines = [line.strip() for line in fin if line.strip()]
        total_lines = len(lines)
        log(f"共读取到 {total_lines} 条有效数据")

    # 处理并写入结果
    with open(args.output, "w", encoding="utf-8") as fout:
        for line_num, line in enumerate(tqdm(lines, desc="处理进度"), 1):
            try:
                # 解析JSON
                data = json.loads(line)
                
                # 检查并替换image字段
                if "image" in data:
                    original_image = data["image"]
                    # 处理可能的列表格式（如果image是列表）
                    if isinstance(data["image"], list):
                        data["image"] = [
                            replace_image_path(img, new_str=args.replace_with) 
                            for img in data["image"]
                        ]
                    else:
                        data["image"] = replace_image_path(data["image"], new_str=args.replace_with)
                    success_lines += 1
                else:
                    error_lines.append(f"行 {line_num}：缺少'image'字段，跳过处理")
                    continue

                # 写入修改后的数据
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")

            except json.JSONDecodeError:
                error_lines.append(f"行 {line_num}：JSON格式错误，无法解析")
            except Exception as e:
                error_lines.append(f"行 {line_num}：处理失败 -> {str(e)}")

    # 输出处理结果
    log("\n===== 处理完成 =====")
    log(f"总数据量：{total_lines}")
    log(f"成功处理：{success_lines}")
    log(f"处理失败：{len(error_lines)}")
    log(f"输出文件：{args.output}")

    # 打印前5条错误（如果有）
    if error_lines and len(error_lines) > 0:
        log("\n前5条错误信息：")
        for err in error_lines[:5]:
            log(f"- {err}")


if __name__ == "__main__":
    # 使用示例：
    # python replace_image_path.py --output ./modified.jsonl --replace-with "my_custom_dir"
    main()