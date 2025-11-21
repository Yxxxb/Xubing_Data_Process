import requests
import json
from PIL import Image
from datakit.utils.image import encode_pil_to_base64_image

url = "http://localhost:8081/v1/chat/completions"

with open('points-1.2/prompts/reasoning_general_qa.txt') as f:
    PROMPT = f.read()

with open('test.txt') as f:
    conversation_str = f.read()


def obtain_results(i):
    pil_image = Image.open('/mnt/cephfs/bensenliu/code/wepoints_series/image.png')
    base64_image = encode_pil_to_base64_image(pil_image)
    data = {
        "model": "Qwen/Qwen2-VL-7B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    }
                ],
            }
        ],
        "max_tokens": 1024,
    }
    response = requests.post(url, json=data)
    response = json.loads(response.text)
    response = response['choices'][0]['message']['content']
    return  response


if __name__ == '__main__':
    response = obtain_results(0)
    print(response)
