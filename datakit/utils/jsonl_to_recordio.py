import os
from typing import List

from tqdm import tqdm

from .distributed import (barrier_all_processes, dist_split_files,
                          get_distributed_env, gpu_utilization, kill_process)
from .files import find_all_files


def jsonl_to_recordio(
        root_list: str,
        alert_path: str = '/mnt/cephfs/haichengwang/envs/loop_test',
        num_jsonls_per_shard: str = 400,
        is_pretrain_decay: bool = False) -> List[str]:
    """
    Convert jsonl files to recordio files.

    Args:
        root_list (str): Root directory of the jsonl files.
        alert_path (str): Path to the alert file.
                        Defaults to '/mnt/cephfs/haichengwang/envs/loop_test'.
        num_jsonls_per_shard (int): Number of jsonl files per shard.
                                    Defaults to 400.

    Returns:
        list: List of paths to the recordio files.
    """
    gpu_utilization()
    root_list_ori = root_list.split(',')

    root_list = []
    for root in tqdm(root_list_ori):
        # find all first level directories under root
        root = root.strip()
        dirs = os.listdir(root)
        only_file = True
        for dir in tqdm(dirs):
            dir_path = os.path.join(root, dir)
            if os.path.isdir(dir_path):
                root_list.append(dir_path)
                only_file = False
        if only_file:
            root_list.append(root)

    output_root_list = []

    for root in tqdm(root_list):
        if '/jsonl' not in root:
            print(f'{root} is not a jsonl directory.')
            continue
        output_root = root.replace('/jsonl/', '/recordio/')
        output_root_list.append(output_root)
        os.makedirs(output_root, exist_ok=True)
        all_jsonls = find_all_files(root, 'jsonl')
        jsonls_cur_rank = dist_split_files(all_jsonls)
        world_size, rank, _ = get_distributed_env()
        num_slice_cur_rank = len(jsonls_cur_rank) // num_jsonls_per_shard + 1
        for i in range(num_slice_cur_rank):
            cur_jsonls = jsonls_cur_rank[i * num_jsonls_per_shard:(i + 1) *
                                         num_jsonls_per_shard]
            jsonl_str = ' '.join(cur_jsonls)
            output_data = os.path.join(output_root,
                                       f'{rank}_{i}_data.recordio')
            image_data = os.path.join(output_root,
                                      f'{rank}_{i}_image.recordio')
            if not is_pretrain_decay:
                # instruct
                cmd = ('mmq_prepare_multi_modal_data '
                       f'--output_file {output_data} '
                       f'--output_image_file {image_data} '
                       '--tokenizer.type huggingface '
                       '--tokenizer.vocab_file /mnt/cephfs/bensenliu/wfs/weights/nlp/Qwen2.5-7B-Instruct '
                       f'--input_files {jsonl_str} '
                       "--role_mapping $'user=<|im_start|>user\n,assistant=<|im_start|>assistant\n,system=<|im_start|>system\n' "
                       '--concurrency 200 '
                       '--chunk_size 1024 '
                       '--num_modal_tokens 144 '
                       "--eot_token '<|im_end|>' "
                       '--special_masked_token "<|image_pad|>" '
                       '--mask_user '
                       '--add_eot false')
            else:
                # pt 续写
                cmd = ('mmq_prepare_multi_modal_data '
                    f'--output_file {output_data} '
                    f'--output_image_file {image_data} '
                    '--tokenizer.type huggingface '
                    f'--tokenizer.vocab_file /mnt/cephfs/bensenliu/wfs/weights/nlp/Qwen2.5-7B-Instruct '
                    f'--input_files {jsonl_str} '
                    '--concurrency 10 '
                    '--chunk_size 1024 '
                    '--num_modal_tokens 144 '
                    "--eot_token '<|im_end|>' "
                    '--special_masked_token "<|image_pad|>" '
                    '--add_eot false '
                    '--no_role')
            os.system(cmd)
        print('Finished!')

    barrier_all_processes(task_name='recordio',
                          root=alert_path)  # wait for all processes to finish
    kill_process()  # kill max GPU utilization process
    return output_root_list
