import subprocess
import os
from tqdm import tqdm
from datakit.utils.distributed import (dist_split_files,
                                       get_distributed_env)
from datakit.utils.files import (find_all_files)
from data_distribution import data_distribution
from video_data_distribution import video_data_distribution


is_video = False
SUBSET = 'grammar_correct'
include_datasets = [
    # 'LLaVA-Video-178K',
    # 'ShareGPT4Video',
    'belle_cn_1m'
]
if is_video:
    root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category-video'
    data_structure = video_data_distribution
else:
    root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category'
    data_structure = data_distribution
points = []
for category, dataset_names in data_structure.items():
    for dataset_name in dataset_names:
        if dataset_name not in include_datasets:
            continue
        points.append(f'{category}/{dataset_name}')


if __name__ == '__main__':
    filenames_cur_rank = dist_split_files(points)
    world_size, rank, _ = get_distributed_env()
    success_data = []
    failed_data = []
    for filename in tqdm(filenames_cur_rank):
        print(f"Processing {filename}")
        file_root = os.path.join(
            root, filename, f'jsonl/qwen2vl-{SUBSET}')
        data_name = filename.split('/')[1]
        output_root = os.path.join(
            root, filename, f'recordio/qwen2vl-{SUBSET}-welm-fix')
        os.makedirs(output_root, exist_ok=True)
        files = find_all_files(file_root, 'jsonl')
        if len(files) == 0:
            print(f"No files found in {file_root}")
            failed_data.append(filename)
            continue
        files_str = ' '.join(files)
        output_data = os.path.join(output_root, f'{data_name}_data.recordio')
        image_data = os.path.join(output_root, f'{data_name}_image.recordio')
        cmd = (
            'mmq_prepare_multi_modal_data '
            f'--output_file {output_data} '
            f'--output_image_file {image_data} '
            '--tokenizer.type huggingface '
            '--tokenizer.vocab_file /mnt/wfs/mmnanjingwfssh/project_pr-nlp-large_pretrain/zilinzhu/auto_sft/tokenizer/welm '
            f'--input_files {files_str} '
            "--role_mapping $'user= <end_of_turn>user\n,assistant= <end_of_turn>assistant\n,system= <end_of_turn>system\n' "
            '--concurrency 200 '
            '--chunk_size 1024 '
            '--num_modal_tokens 144 '
            "--eot_token ' <end_of_turn>' "
            '--special_masked_token " <extra_id_0>" '
            '--mask_user '
            '--add_eot false'
        )
        print(cmd)
        result = subprocess.run(
            cmd, shell=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode == 0:
            success_data.append(filename)
        else:
            failed_data.append(filename)
            print(result.stdout)
    print(f"Success data: {success_data}")
    print(f"Failed data: {failed_data}")
