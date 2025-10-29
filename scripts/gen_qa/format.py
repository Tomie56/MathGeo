import json
import argparse

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="格式化JSONL数据并筛选gt.expr过长的条目")
    parser.add_argument("input_path", help="输入JSONL文件路径")
    parser.add_argument("--output", required=True, help="输出JSONL文件路径")
    parser.add_argument("--max-expr-length", type=int, default=30, 
                        help="gt.expr的最大允许字符数（超过则跳过，默认200）")
    args = parser.parse_args()

    # 处理数据
    with open(args.input_path, 'r', encoding='utf-8') as fin, \
         open(args.output, 'w', encoding='utf-8') as fout:
        
        total = 0
        kept = 0
        skipped = 0

        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                data = json.loads(line)
                
                # 检查必要字段是否存在
                if "gt" not in data:
                    print(f"跳过缺少'gt'字段的条目（行号：{total}）")
                    skipped += 1
                    continue
                gt = data["gt"]
                if "expr" not in gt:
                    print(f"跳过'gt'中缺少'expr'字段的条目（行号：{total}）")
                    skipped += 1
                    continue

                # 筛选：如果expr长度超过阈值则跳过
                expr = gt["expr"]
                if len(expr) > args.max_expr_length:
                    print(f"跳过expr过长的条目（长度：{len(expr)}，阈值：{args.max_expr_length}，行号：{total}）")
                    skipped += 1
                    continue

                # 构建新格式数据（保持原有逻辑）
                new_data = {
                    "image": data.get("annotated_raw_path", "").replace("./", ""),  # 处理路径
                    "question": data.get("question", ""),
                    "qt_description": data.get("qt_description", ""),
                    "description": data.get("description", ""),
                    "gt": gt,
                    "level": data.get("level", "unknown")
                }

                # 写入输出文件
                fout.write(json.dumps(new_data, ensure_ascii=False) + "\n")
                kept += 1

            except json.JSONDecodeError:
                print(f"跳过JSON解析错误的行（行号：{total}）")
                skipped += 1
                continue
            except Exception as e:
                print(f"处理行号{total}时出错：{str(e)}，已跳过")
                skipped += 1
                continue

    # 输出统计信息
    print(f"处理完成：总条目{total}，保留{kept}，跳过{skipped}")
    print(f"结果已保存至：{args.output}")

if __name__ == "__main__":
    main()