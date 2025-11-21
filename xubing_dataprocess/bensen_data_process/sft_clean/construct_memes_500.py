from datasets import load_dataset
from datakit.utils.files import dump_list_to_jsonl_file
from datakit import Qwen2VLLMWrapper
from datakit.utils.image import encode_pil_to_base64_image
import random
from tqdm import tqdm
import os


PROMPTS = [
    'What occasions would someone use this meme?'
    'Can you explain this meme?',
    'What is funny about this image?'
]

model_path = '/mnt/cephfs/bensenliu/wfs/weights/mm/opensource/qwen2-vl-72b-instruct'
model = Qwen2VLLMWrapper(model_path, max_tokens=1024,
                         tensor_parallel_size=8)


if __name__ == '__main__':
    data_file = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/memes-500'
    output_file = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/memes-500/data/grammar_correct/memes_500.jsonl'
    dataset = load_dataset(data_file)['train']
    results = []
    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        pil_image = item['image']
        base64_image = encode_pil_to_base64_image(pil_image)
        pil_image.save('temp.jpg')
        prompt = random.choice(PROMPTS)
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
                'content': '\n### 问题\nWhat is funny about this image?\n### 答案\nIt is a cartoon of a rhinoceros painting a picture and each picture has its rhino horn because the rhino horn grows in front of its eyes. The caption "You see the world as you are!" is a playful commentary on how our perspective shapes our perception of the world.\n\n'
            },
            {
                'type': 'text',
                'content': '## 任务\n### 图片\n'
            },
            {
                'type': 'image',
                'content': './temp.jpg'
            },
            {
                'type': 'text',
                'content': f'\n### 问题\n{prompt}\n### 答案\n'
            }
        ]
        answer = model.generate(messages)
        template = {
            'id': str(i),
            'base64_image': {
                str(i): base64_image
            },
            'conversations': [
                {'role': 'user', 'messages': prompt},
                {'role': 'assistant', 'messages': answer}
            ]
        }
        results.append(template)
        os.remove('temp.jpg')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    dump_list_to_jsonl_file(output_file, results)