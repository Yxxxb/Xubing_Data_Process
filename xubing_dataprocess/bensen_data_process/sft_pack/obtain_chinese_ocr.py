from datakit.utils.files import read_jsonl_file, dump_list_to_jsonl_file
from datakit.utils.distributed import dist_split_files, get_distributed_env
from datakit.utils.image import save_base64_image, decode_base64_image_to_pil, encode_pil_to_base64_image
from datakit import Qwen2VLLMWrapper
from tqdm import tqdm
import random
import os


model_path = '/mnt/cephfs/bensenliu/wfs/weights/mm/opensource/qwen2-vl-72b-instruct'
model = Qwen2VLLMWrapper(model_path, max_tokens=2048,
                         tensor_parallel_size=8)

with open('points-1.2/prompts/ocr_cn_oneshot_sample.txt', encoding='utf-8') as f:
    answer_sample = f.read()


PROMPTS = [
    "请提取图片中的所有文字",
    "请从图片中提取文字",
    "请从图片中提取全部文字",
    "请从图片中提取出所有文字",
    "请提取所有图片中的文字",
    "请从图片中获取所有文字",
    "这张图片里的文字是什么",
    "这幅图上的文字内容是什么",
    "请问图片中的文字是什么",
    "你能告诉我这张图片上的文字是什么吗",
    "这张图片里写了什么",
    "图片上显示的文字是什么",
    "Please extract all the text from the image",
    "Please extract text from the image",
    "Please extract all the text from the image",
    "Please extract all the text from the image",
    "Please extract the text from all the images",
    "Please get all the text from the image",
    "What is the text in this image",
    "What is the text content on this picture",
    "What is the text in the image",
    "Can you tell me what the text on this image is",
    "What is written in this image",
    "What is the text displayed on the image"
]


def resize_image_to_max_size(image, max_pixels):
    height, width = image.size
    ratios = (height * width) / max_pixels
    height = int(height / ratios ** 0.5)
    width = int(width / ratios ** 0.5)
    image = image.resize((height, width))
    return image


if __name__ == '__main__':
    _, rank, _ = get_distributed_env()
    input_file = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/Chinese-OCR/data/Chinese-OCR.jsonl'
    output_file = f'/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/Chinese-OCR/data/grammar_correct/Chinese-OCR_{rank}.jsonl'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    data = read_jsonl_file(input_file)
    data_cur_rank = dist_split_files(data)
    results = []
    for i, base64_image in enumerate(tqdm(data_cur_rank)):
        pil_image = decode_base64_image_to_pil(base64_image)
        pil_image = resize_image_to_max_size(pil_image, 2048 * 2048)
        base64_image = encode_pil_to_base64_image(pil_image)
        save_base64_image(base64_image, f'temp_cn_ocr_{rank}.jpg')
        sample_id = input_file + f'_{i}'
        prompt = random.choice(PROMPTS)
        messages = [
            {
                'type': 'text',
                'content': '现在有一张图片和一个问题，请根据问题回答。请仿照下面的例子回答:\n## 例子\n### 图片\n'
            },
            {
                'type': 'image',
                'content': './image.webp'
            },
            {
                'type': 'text',
                'content': f'\n### 问题\n请提取出图片里包含的文字\n### 答案\n{answer_sample}\n\n'
            },
            {
                'type': 'text',
                'content': '## 任务\n### 图片\n'
            },
            {
                'type': 'image',
                'content': f'./temp_cn_ocr_{rank}.jpg'
            },
            {
                'type': 'text',
                'content': prompt
            }
        ]
        answer = model.generate(messages)
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
    os.remove(f'temp_cn_ocr_{rank}.jpg')
    dump_list_to_jsonl_file(output_file, results)
    print(f'Rank {rank} finish.')
    




