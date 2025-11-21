from tqdm import tqdm
from datakit.utils.distributed import (dist_split_files,
                                       get_distributed_env,)
from datakit.utils.files import read_jsonl_file, dump_list_to_jsonl_file, find_all_files
from datakit.utils.mp import multi_process_with_append
import os

points = [
    'alpaca-gpt4',
    'openhermes2.5',
    'mini-gemini',
    # 'docvqa',
    # 'icdar_2015',
    # 'MathV360K',
    # 'sharegpt4v_llava158k',
    # 'sharegpt4v_qa',
    # 'sharegpt4v_sharegpt'
]


def filter_single_item(item):
    if item['type'] == 'image':
        return item
    conversations = item['conversations']
    for i in range(len(conversations)//2):
        answer = conversations[2*i+1]['text']
        if answer.endswith('#'):
            return None
    return item


if __name__ == '__main__':
    filenames_cur_rank = dist_split_files(points)
    world_size, rank, _ = get_distributed_env()
    success_data = []
    failed_data = []
    for filename in tqdm(filenames_cur_rank):
        try:
            print(f"Processing {filename}")
            data_file = f'/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/{filename}/jsonl/qwen2vl-grammar-correct'
            output_file = f'/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/{filename}/jsonl/qwen2vl-grammar-correct-hashtag-removed/{filename}.jsonl'
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            data_file = find_all_files(data_file, 'jsonl')[0]
            data = read_jsonl_file(data_file)
            outlier_data = multi_process_with_append(filter_single_item, data, 256)
            print(f"Found {len(outlier_data)} outliers in {filename}")
            dump_list_to_jsonl_file(output_file, outlier_data)
            success_data.append(filename)
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
            failed_data.append(filename)
    print(f"Success: {success_data}")
    print(f"Failed: {failed_data}")
        

        
