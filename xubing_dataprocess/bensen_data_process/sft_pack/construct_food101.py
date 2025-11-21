from datakit.utils.files import dump_list_to_jsonl_file, read_jsonl_file
import os
import re
from collections import Counter
import random
from tqdm import tqdm
from datakit.utils.image import save_base64_image
from datakit import Qwen2VLLMWrapper
from datakit.utils.distributed import dist_split_files, get_distributed_env


PROMPTS = [
    "What is the name of this dish?",
    "How to cook this dish?"
]

with open('points-1.2/prompts/dish.txt') as f:
    answer_sample = f.read()

model_path = '/mnt/cephfs/bensenliu/wfs/weights/mm/opensource/qwen2-vl-72b-instruct'
model = Qwen2VLLMWrapper(model_path, max_tokens=1024,
                         tensor_parallel_size=8)


def word_frequency(text):
    words = re.findall(r'\b\w+\b', text.lower())
    word_counts = Counter(words)
    values = list(word_counts.values())
    if max(values) > 10:
        return False
    return True


if __name__ == '__main__':
    _, rank, _ = get_distributed_env()
    input_file = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/food101-food-chinese/data/food101-food-chinese.jsonl'
    output_file = f'/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/food101-food-chinese/data/grammar_correct/food101-food-chinese_{rank}.jsonl'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    data = read_jsonl_file(input_file)
    random.shuffle(data)
    data_cur_rank = dist_split_files(data)
    results = []
    for i, item in enumerate(tqdm(data_cur_rank)):
        base64_image = item
        save_base64_image(base64_image, f'temp_{rank}.jpg')
        prompt = random.choice(PROMPTS)
        sample_id = f'{input_file}_{i}'
        messages = [
                        {
                            'type': 'text',
                            'content': '现在有一张图片和一个问题，请根据问题回答。请仿照下面的例子回答:\n## 例子\n### 图片\n'
                        },
                        {
                            'type': 'image',
                            'content': './image.png'
                        },
                        {
                            'type': 'text',
                            'content': f'\n### 问题\nHow to cook this dish?\n### 答案\n{answer_sample}\n\n'
                        },
                        {
                            'type': 'text',
                            'content': '## 任务\n### 图片\n'
                        },
                        {
                            'type': 'image',
                            'content': f'./temp_{rank}.jpg'
                        },
                        {
                            'type': 'text',
                            'content': f'\n### 问题\n{prompt}\n### 答案\n'
                        }
                ]
        answer = model.generate(messages)
        if "don't" in answer.lower() or 'sorry' in answer.lower() or "can't" in answer.lower() or "no text" in answer.lower() or 'instructions' in answer.lower():
            continue
        if not word_frequency(answer):
            continue
        print(answer)
        template = {
            'id': sample_id,
            'base64_image': {
                sample_id: base64_image
            },
            'conversations': [
                {'role': 'user', 'text': prompt},
                {'role': 'assistant', 'text': answer}
            ]
        }
        results.append(template)
    os.remove(f'temp_{rank}.jpg')
    dump_list_to_jsonl_file(output_file, results)


