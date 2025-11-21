# 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
# 上面是我的openai key，我现在需要调用这个key来调用gpt5模型，并返回结果

import openai

openai.api_key = "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"

response = openai.ChatCompletion.create(
    model="gpt-5",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ],
    max_completion_tokens=1000
)
print(response.choices[0].message.content)