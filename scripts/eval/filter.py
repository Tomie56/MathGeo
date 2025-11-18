import json

# 输入和输出路径
input_path = "/mnt/afs/jingjinhao/project/GeoChain/MathGeo/A500_v3/output/A500_v3_level.jsonl"
output_path = "/mnt/afs/jingjinhao/project/GeoChain/MathGeo/A500_v3/output/A500_v3_level_judged.jsonl"

# 读取并筛选数据
filtered = []
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        if item.get("judge") is True:
            filtered.append(item)

# 按 level 从大到小排序
# filtered.sort(key=lambda x: x.get("level", 0), reverse=True)

# 保存结果
with open(output_path, "w", encoding="utf-8") as f:
    for item in filtered:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ 已筛选并保存 {len(filtered)} 条 judge=true 的数据到：{output_path}")
