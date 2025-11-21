from datakit.utils.files import find_all_files, dump_list_to_jsonl_file
from datakit.utils.image import encode_bytes_to_base64_image
from datakit.utils.distributed import dist_split_files, get_distributed_env
from datakit.utils.mp import multi_process_with_append
import pandas as pd
import random
import os
from tqdm import tqdm


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
    byte_image = line['image']['bytes']
    base64_image = encode_bytes_to_base64_image(byte_image)
    caption = eval(line['ground_truth'])['gt_parse']['text_sequence']
    prompt = random.choice(OCR_PROMPTS)
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
    world_size, rank, locol_rank = get_distributed_env()
    input_root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/pt/synthdog-en/data'
    output_file = f'/mnt/cephfs/bensenliu/wfs/vlmdatasets/pt/synthdog-en/data/sft-format/{rank}.jsonl'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    all_parquets = find_all_files(input_root, 'parquet')
    all_parquets = [p for p in all_parquets if 'train' in p]
    all_parquets = dist_split_files(all_parquets)
    all_lines = []
    for parquet in tqdm(all_parquets):
        df = pd.read_parquet(parquet)
        lines = [line.to_dict() for _, line in df.iterrows()]
        all_lines.extend(lines)
    all_lines = [[output_file+str(i), line] for i, line in enumerate(all_lines)]
    results = multi_process_with_append(process_single_line, all_lines, num_workers=64)
    dump_list_to_jsonl_file(output_file, results)
    print(f'Done! Output file: {output_file}')

