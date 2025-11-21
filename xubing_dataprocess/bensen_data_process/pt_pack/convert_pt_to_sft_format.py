from datakit.utils.files import (find_all_files, 
                                 read_jsonl_file,
                                 dump_list_to_jsonl_file)
from datakit.utils.distributed import dist_split_files
from datakit.utils.mp import multi_process_with_append
import random
from tqdm import tqdm
import os



PROMPTS = [
    "could you describe this image?",
    "can you describe this image?",
    "would you describe this image?",
    "please provide a description of this image.",
    "please give a description of this image.",
    "please explain this image.",
    "please tell me about this image.",
    "please detail this image.",
    "please describe what is in this image.",
    "please describe the contents of this image."
]

OCR_PROMPTS = [
    "Could you extract all the text from the image?",
    "Can you extract all the text from the image?",
    "Would you extract all the text from the image?",
    "Please retrieve all the text from the image.",
    "Please get all the text from the image.",
    "Please pull all the text from the image.",
    "Please obtain all the text from the image.",
    "Please take out all the text from the image.",
    "Please extract every piece of text from the image.",
    "Please extract the entire text from the image."
]


def process_single_line(items):
    sample_id, line = items
    base64_image = line['base64_image']
    ocr = line['ocr'] if 'ocr' in line else None
    if ocr is not None and len(ocr) > 60:
        caption = ocr
        prompt = random.choice(OCR_PROMPTS)
    else:
        caption = line['internlm2_xcomposer2_caption']
        prompt = random.choice(PROMPTS)
    if len(caption) < 10:
        return None
    reformat_line = {
        'id': sample_id,
        'base64_image': {
            sample_id: base64_image
        },
        'conversations': [
            {
                'role': 'user',
                'text': prompt
            },
            {
                'role': 'assistant',
                'text': caption
            }
        ]
    }
    return reformat_line


if __name__ == '__main__':
    root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/pt/laion5b-en-512/data/1m-ppl-internvl8b'
    sft_reformat_root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/pt/laion5b-en-512/data/1m-ppl-internvl8b-sft'
    all_jsonls = find_all_files(root, 'jsonl')
    jsonls_cur_rank = dist_split_files(all_jsonls)
    for jsonl in tqdm(jsonls_cur_rank):
        subfolder = jsonl.split('/')[-2]
        filename = jsonl.split('/')[-1]
        output_file = os.path.join(sft_reformat_root, subfolder, filename)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        data = read_jsonl_file(jsonl)
        packed_data = [[output_file + '-' + str(i), line] 
                       for i, line in enumerate(data)]
        results = multi_process_with_append(process_single_line, packed_data, 64)
        dump_list_to_jsonl_file(output_file, results)
    print('Done!')
