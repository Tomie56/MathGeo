import json

# 输入文件路径
input_path = "/mnt/afs/jingjinhao/project/GeoChain/MathGeo/A500_v4/output/all_test_answer_level_shuffled_judged.jsonl"
# 输出文件路径（避免覆盖原文件）
output_path = "/mnt/afs/jingjinhao/project/GeoChain/MathGeo/A500_v4/output/A500_v4_no_system.jsonl"

with open(input_path, 'r', encoding='utf-8') as f_in, \
     open(output_path, 'w', encoding='utf-8') as f_out:
    
    for line in f_in:
        try:
            data = json.loads(line.strip())
            # 获取 conversations 数组（若不存在则跳过）
            conversations = data.get("conversations", [])
            system_message = {
                "from": "system",
                "value": "Reason step by step and place the thought process within the <think></think> tags, and provide the final conclusion at the end."
            }
            # conversations.insert(0, system_message) 
            # 遍历数组中的每个对话对象
            for conv in conversations:
                # 检查是否是 human 的对话且包含 value 字段
                if conv.get("from") == "human" and "value" in conv:
                    # 在 value 开头添加 <image>
                    conv["value"] = f"<image>\n{conv['value']}"
                if conv.get("from") == "gpt" and "value" in conv:
                    
                    conv["value"] = f"<think>\n{conv['value']}"
            # 写入修改后的数据
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
        except json.JSONDecodeError:
            # 保留解析错误的原始行
            f_out.write(line)
            print(f"解析错误，已保留原行：{line.strip()}")

print(f"处理完成，结果已保存至：{output_path}")