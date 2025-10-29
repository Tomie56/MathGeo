import json
import re
import os

# 输入文件路径
input_path = "/mnt/afs/jingjinhao/project/GeoChain/MathGeo/scripts/call_api/answer.jsonl"
output_path = "/mnt/afs/jingjinhao/project/GeoChain/MathGeo/scripts/call_api/final.jsonl"

# 正则表达式：匹配 \boxed{...} 中的内容（支持嵌套花括号）
BOXED_PATTERN = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'

def extract_boxed_content(text):
    """从文本中提取 \boxed{...} 中的内容"""
    match = re.search(BOXED_PATTERN, text)
    if match:
        return match.group(1).strip()
    return None

def process_line(line):
    data = json.loads(line.strip())
    
    # 提取 api_results 内容（通常为列表，取第一个元素）
    api_text = data.get("api_results", [""])[0] if isinstance(data.get("api_results"), list) else ""
    
    # 提取 answer 字段中的 boxed 内容
    answer_text = data.get("answer", [""])[0] if isinstance(data.get("answer"), list) else ""
    boxed_value = extract_boxed_content(answer_text)
    image_path = data.get("image", "")
    
    # 构造新结构
    new_item = {
        "image": image_path, 
        "width": None,
        "height": None,
        "conversations": [
            {
                "from": "human",
                "value": f"<image>{api_text}"
            },
            {
                "from": "gpt",
                "value": answer_text 
            }
        ],
        "answer": boxed_value, 
        "gt": data.get("gt", {}) 
    }
    
    return new_item

# 主处理逻辑
with open(input_path, 'r', encoding='utf-8') as fin, \
     open(output_path, 'w', encoding='utf-8') as fout:

    for line in fin:
        try:
            processed = process_line(line)
            fout.write(json.dumps(processed, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Error processing line: {line[:100]}... -> {e}")

print(f"✅ 处理完成！输出文件已保存至: {output_path}")