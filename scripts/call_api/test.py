import os
from openai import OpenAI

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key="sk-118c7a1f568f42ee93bc0cdf5b95fe17",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
completion = client.chat.completions.create(
    model="qwen3-vl-235b-a22b-thinking",  # 此处以qwen-vl-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    messages=[{"role": "user","content": [
            {"type": "text", "text": "这是什么"},
            ]}],
    temperature=0.6,
    max_tokens=32768,
    top_p=0.95,
    )
print(completion.model_dump_json())