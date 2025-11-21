from datakit import query_qwen2vl
from datakit.utils.files import read_jsonl_file, dump_list_to_jsonl_file
from datakit.utils.mp import multi_process_with_append
from datakit.utils.distributed import dist_split_files, get_distributed_env
from datakit.prompts import OCR_PROMPTS
from datakit.utils.image import save_base64_image
import random
import os
import re


root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/OCR'
PATTERN = r'```(.*?)```'


def extract_single_item(item):
    try:
        base64_images = item['base64_image']
        image_paths = []
        for base64_image_id, base64_image in base64_images.items():
            image_path = f'/tmp/{base64_image_id}.jpg'
            save_base64_image(base64_image, image_path)
            image_paths.append(image_path)
        prompt = '请提取出图片中的文字'
        answer = query_qwen2vl(image_paths, prompt, 2048)
        match = re.search(PATTERN, answer, re.DOTALL)
        if match:
            answer = match.group(1).strip()
        if len(answer) > 1500:
            print(f'answer too long: {answer}')
            return None
        conversations = [
            {
                'role': 'user',
                'text': random.choice(OCR_PROMPTS)
            },
            {
                'role': 'assistant',
                'text': answer
            }
        ]
        item['conversations'] = conversations
        for image_path in image_paths:
            os.remove(image_path)
    except Exception as e:
        print(f'Error in extract_single_item: {e}')
        return None
    return item


if __name__ == '__main__':
    _, rank, _ = get_distributed_env()
    data_path = f'{root}/card_certificate_ticket_cn/data/grammar_correct/card_certificate_ticket_cn.jsonl' 
    output_path = f'{root}/card_certificate_ticket_cn/data/grammar_correct_dist/card_certificate_ticket_cn_{rank}.jsonl' 
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = read_jsonl_file(data_path)
    data_cur_rank = dist_split_files(data)
    results = multi_process_with_append(extract_single_item, data_cur_rank, 
                                        num_workers=16)
    dump_list_to_jsonl_file(output_path, results)
    print('All done!')