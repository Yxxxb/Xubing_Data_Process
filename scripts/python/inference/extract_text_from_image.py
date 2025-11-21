import argparse
import os
import random
import re

from tqdm import tqdm

from datakit import Qwen2VLLMWrapper
from datakit.prompts import OCR_PROMPTS
from datakit.utils.distributed import dist_split_files, get_distributed_env
from datakit.utils.files import dump_list_to_jsonl_file, read_jsonl_file
from datakit.utils.image import (decode_base64_image_to_pil,
                                 encode_pil_to_base64_image,
                                 resize_image_to_max_size,
                                 rotate_images_in_item, save_base64_image)

parser = argparse.ArgumentParser()
parser.add_argument('--input-file', type=str, required=True)
parser.add_argument('--output-folder', type=str, required=True)
parser.add_argument('--model-path', type=str, required=True)
parser.add_argument('--max-pixels', type=int, default=2048 * 2048)
parser.add_argument('--tensor-parallel-size', type=int, default=8)

PATTERN = r'```(.*?)```'

if __name__ == '__main__':
    _, rank, _ = get_distributed_env()
    args = parser.parse_args()
    model = Qwen2VLLMWrapper(args.model_path,
                             max_tokens=2048,
                             tensor_parallel_size=args.tensor_parallel_size)
    input_file = args.input_file
    output_folder = args.output_folder
    max_pixels = args.max_pixels
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f'results_{rank}.jsonl')
    data = read_jsonl_file(input_file)
    data_cur_rank = dist_split_files(data)
    results = []
    for item in tqdm(data_cur_rank):
        base64_images = item['base64_image']
        base64_images_list = []
        prompt = random.choice(OCR_PROMPTS)
        for _, base64_image in base64_images.items():
            pil_image = decode_base64_image_to_pil(base64_image)
            pil_image = resize_image_to_max_size(pil_image, max_pixels)
            base64_image = encode_pil_to_base64_image(pil_image)
            base64_images_list.append(base64_image)
        for i, base64_image in enumerate(base64_images_list):
            save_base64_image(base64_image, f'tmp_{i}_{rank}.jpg')
        messages = [{
            'type': 'image',
            'content': f'tmp_{i}_{rank}.jpg'
        } for i in range(len(base64_images_list))]
        messages.append({'type': 'text', 'content': prompt})
        answer = model.generate(messages)
        match = re.search(PATTERN, answer, re.DOTALL)
        if match:
            answer = match.group(1).strip()
        print(answer)
        base64_images_dict = {
            str(hash(base64_image)): base64_image
            for base64_image in base64_images_list
        }
        item['conversations'] = [{
            'role': 'user',
            'text': prompt
        }, {
            'role': 'assistant',
            'text': answer
        }]
        item['base64_image'] = base64_images_dict
        for i in range(len(base64_images_list)):
            os.remove(f'tmp_{i}_{rank}.jpg')
        item_list = rotate_images_in_item(item, [90, 180, 270])
        results.extend(item_list)
    dump_list_to_jsonl_file(output_file, results)
