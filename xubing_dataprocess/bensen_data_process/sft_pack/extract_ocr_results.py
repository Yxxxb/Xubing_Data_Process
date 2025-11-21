from datakit import query_qwen2vl
from datakit.utils.files import read_jsonl_file, dump_list_to_jsonl_file
from datakit.utils.mp import multi_process_with_append
from datakit.utils.distributed import dist_split_files, get_distributed_env
from datakit.prompts import OCR_PROMPTS
from datakit.utils.image import save_base64_image
import random
import os
import re

datasets = [
    'aiplane-itinerary-cn',
    'bank_card_cn',
    'boarding_pass_cn',
    'bus_ticket_cn',
    'customs_declaration_cn',
    'duty-paid-proof-cn',
    'express_waybill_cn',
    'invoice-mix-up-cn',
    'medical_bill_cn',
    'quota_invoice_cn',
    'railway_ticket_cn',
    'ride_sharing_cn',
    'steamer_ticket_cn',
    'taxi_ticket_cn',
    'toll_invoice_cn',
    'vat_invoice_roll_cn'
]

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
    datasets_cur_rank = dist_split_files(datasets)
    print(f'rank {rank} start processing {datasets_cur_rank}')
    for dataset in datasets_cur_rank:
        print(f'Processing {dataset}...')
        data_path = f'{root}/{dataset}/data/grammar_correct/{dataset}.jsonl'
        data = read_jsonl_file(data_path)
        results = multi_process_with_append(extract_single_item, data, 
                                            num_workers=16)
        dump_list_to_jsonl_file(data_path, results)
        print(f'Finished processing {dataset}!')
    print('All done!')