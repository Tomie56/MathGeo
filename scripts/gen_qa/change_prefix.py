import json
import os

def replace_results_in_paths(jsonl_path: str, target_str: str, output_path: str = None):
    """
    替换JSONL文件中 raw_path 和 annotated_raw_path 字段的 "results" 为指定字符串
    
    Args:
        jsonl_path: 输入JSONL文件路径（绝对路径）
        target_str: 替换 "results" 的指定字符串
        output_path: 输出文件路径（默认生成带"modified"后缀的新文件）
    """
    # 校验输入文件是否存在
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"输入文件不存在：{jsonl_path}")
    
    # 配置输出路径
    if not output_path:
        dir_name = os.path.dirname(jsonl_path)
        file_name = os.path.basename(jsonl_path)
        name_without_ext, ext = os.path.splitext(file_name)
        output_path = os.path.join(dir_name, f"{name_without_ext}_modified{ext}")
    
    # 逐行处理JSONL
    modified_count = 0
    with open(jsonl_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                f_out.write("\n")
                continue
            
            try:
                # 解析JSON数据
                data = json.loads(line)
                
                # 替换 raw_path 中的 results
                if "raw_path" in data and isinstance(data["raw_path"], str):
                    data["raw_path"] = data["raw_path"].replace("results", target_str)
                
                # 替换 annotated_raw_path 中的 results
                if "annotated_raw_path" in data and isinstance(data["annotated_raw_path"], str):
                    data["annotated_raw_path"] = data["annotated_raw_path"].replace("results", target_str)
                
                # 写入修改后的数据
                json.dump(data, f_out, ensure_ascii=False)
                f_out.write("\n")
                modified_count += 1
                
            except json.JSONDecodeError as e:
                f_out.write(line + "\n")  # 保留错误行
            except Exception as e:
                f_out.write(line + "\n")  # 保留原始行
    
    print(f"处理完成！共修改 {modified_count} 条数据")
    print(f"修改后文件保存至：{output_path}")
    return output_path

# ------------------------------ 执行示例 ------------------------------
if __name__ == "__main__":
    # 配置参数
    INPUT_JSONL = "/mnt/afs/jingjinhao/project/GeoChain/MathGeo/results_n500_v1/json/qa/qa_shaded_with_gt_20251029_100050_775.jsonl"
    TARGET_STR = "results_n500_v1"  # 替换为你需要的指定字符串
    OUTPUT_JSONL = None  # 默认为生成带modified后缀的新文件，可自定义路径
    
    # 执行替换
    replace_results_in_paths(INPUT_JSONL, TARGET_STR, OUTPUT_JSONL)