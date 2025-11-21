from datakit.utils.files import (read_jsonl_file,
                                 find_all_files,
                                 dump_list_to_jsonl_file,
                                 filterout_repeat_images_for_mmq)
from datakit.utils.distributed import dist_split_files, get_distributed_env
import os
from tqdm import tqdm


points_sft = [
    'mm_ai2d'
]


if __name__ == '__main__':
    sft_cur_rank = dist_split_files(points_sft)
    _, rank, _ = get_distributed_env()
    print(f'Rank {rank} is processing {sft_cur_rank}')
    root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2'
    success_files = []
    fail_files = []
    for dataset in tqdm(sft_cur_rank):
        try:
            cur_dataset_root = os.path.join(root, dataset,
                                            'jsonl/qwen2vl-grammar-correct-remove-empty')
            output_file = os.path.join(
                root, dataset, f'jsonl/qwen2vl-grammar-correct-remove-empty/{dataset}.jsonl')
            files = find_all_files(cur_dataset_root, 'jsonl')
            results = []
            for file in files:
                data = read_jsonl_file(file)
                results.extend(data)
                os.remove(file)
            results = filterout_repeat_images_for_mmq(results)
            dump_list_to_jsonl_file(output_file, results)
            success_files.append(output_file)
        except Exception as e:
            fail_files.append(dataset)
            print(f'Error processing {dataset}: {e}')
    print(f'Rank {rank} success files: {success_files}')
    print(f'Rank {rank} fail files: {fail_files}')
