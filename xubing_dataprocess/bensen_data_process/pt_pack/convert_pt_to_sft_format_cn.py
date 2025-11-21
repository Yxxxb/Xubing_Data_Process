from datakit.utils.files import (find_all_files, 
                                 read_jsonl_file,
                                 dump_list_to_jsonl_file)
from datakit.utils.distributed import dist_split_files
from datakit.utils.mp import multi_process_with_append
import random
from tqdm import tqdm
import os



PROMPTS = [
    "你能描述一下这张图片吗？",
    "你可以描述一下这张图片吗？",
    "你愿意描述一下这张图片吗？",
    "请提供这张图片的描述。",
    "请给出这张图片的描述。",
    "请解释一下这张图片。",
    "请告诉我关于这张图片的内容。",
    "请详细描述这张图片。",
    "请描述一下这张图片里有什么。",
    "请描述一下这张图片的内容。"
]


def process_single_line(items):
    sample_id, line = items
    base64_image = line['base64_image']
    caption = line['internlm2_xcomposer2_caption']
    prompt = random.choice(PROMPTS)
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
    root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/pt/laion5b-cn-512/data/1m-ppl-internvl8b'
    sft_reformat_root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/pt/laion5b-cn-512/data/1m-ppl-internvl8b-sft'
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
