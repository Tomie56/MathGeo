import json
import argparse
import re

def process_qt_description(qt_desc):
    """处理qt_description字段：删除指定区间内容并转换步骤描述为英文"""
    if not qt_desc:
        return ""
    
    # 1. 删除"Geometry finalized: 到by splitting"中间的内容
    # 使用正则保留前后关键词，删除中间部分
    qt_desc = re.sub(
        r'(Geometry finalized:).*(by splitting)',
        r'\1\2',  # 保留Geometry finalized:和by splitting
        qt_desc,
        flags=re.DOTALL  # 让.匹配包括换行符在内的所有字符
    )
    
    # 2. 将中文步骤描述转换为英文
    # 替换"第 X 轮操作"为"Round X"
    qt_desc = re.sub(r'第 (\d+) 轮操作', r'Round \1', qt_desc)
    
    # 替换"（共 Y 步）"为"(total Y step/steps)"（处理单复数）
    def replace_steps(match):
        num = int(match.group(1))
        return f"(total {num} step)" if num == 1 else f"(total {num} steps)"
    qt_desc = re.sub(r'（共 (\d+) 步）', replace_steps, qt_desc)
    
    # 替换"第 Z 步:"为"Step Z:"
    qt_desc = re.sub(r'第 (\d+) 步:', r'Step \1:', qt_desc)
    
    return qt_desc

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="格式化JSONL数据并筛选gt.expr过长的条目")
    parser.add_argument("input_path", help="输入JSONL文件路径")
    parser.add_argument("--output", required=True, help="输出JSONL文件路径")
    parser.add_argument("--max-expr-length", type=int, default=30, 
                        help="gt.expr的最大允许字符数（超过则跳过，默认30）")  # 修复原注释默认值错误
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
                    print(f"跳过expr过长的条目（长度：{len(expr)}，阈值：{args.max_expr_length}，行号：{total}, 表达式：{expr}）")
                    skipped += 1
                    continue

                # 处理qt_description字段
                raw_qt_desc = data.get("qt_description", "")
                processed_qt_desc = process_qt_description(raw_qt_desc)

                # 构建新格式数据
                new_data = {
                    "image": data.get("annotated_raw_path", "").replace("./", ""),  # 处理路径
                    "question": data.get("question", ""),
                    "qt_description": processed_qt_desc,  # 使用处理后的内容
                    "description": data.get("description", ""),
                    "gt": gt,
                    "diff": data.get("diff", "unknown")
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