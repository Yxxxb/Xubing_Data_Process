import subprocess
import os
from datakit.utils.distributed import (dist_split_files,
                                       get_distributed_env)
from datakit.utils.files import find_all_files


if __name__ == '__main__':
    root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/doc_ocr/pt/data/20250313e1'
    output_root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/doc_ocr/pt/recordio/20250313e1'
    os.makedirs(output_root, exist_ok=True)
    jsonls = find_all_files(root, 'jsonl')
    jsonls_cur_rank = dist_split_files(jsonls)
    world_size, rank, _ = get_distributed_env()
    jsonls_cur_rank_str = ' '.join(jsonls_cur_rank)
    output_data = os.path.join(output_root, f'{rank}_data.recordio')
    image_data = os.path.join(output_root, f'{rank}_image.recordio')
    cmd = (
        'mmq_prepare_multi_modal_data '
        f'--output_file {output_data} '
        f'--output_image_file {image_data} '
        '--tokenizer.type huggingface '
        '--tokenizer.vocab_file /mnt/cephfs/bensenliu/wfs/weights/nlp/Qwen2.5-7B-Instruct '
        f'--input_files {jsonls_cur_rank_str} '
        "--role_mapping $'user=<|im_start|>user\n,assistant=<|im_start|>assistant\n,system=<|im_start|>system\n' "
        '--concurrency 200 '
        '--chunk_size 1024 '
        '--num_modal_tokens 144 '
        "--eot_token '<|im_end|>' "
        '--special_masked_token "<|image_pad|>" '
        '--mask_user '
        '--add_eot false'
    )
    print(cmd)
    result = subprocess.run(
        cmd, shell=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode == 0:
        print(f'Rank {rank} finished successfully.')
    else:
        print(f'Rank {rank} failed with error code {result.returncode}.')