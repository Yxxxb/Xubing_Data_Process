import subprocess
import os
from tqdm import tqdm
from datakit.utils.distributed import (dist_split_files,
                                       get_distributed_env)
from datakit.utils.files import (find_all_files)


points = [
    'mm_ai2d',
    'docvqa',
    'dvqa',
    'geoqa+',
    'allava_cap',
    'iconqa_choose_txt',
    'iconqa_fill_blank',
    'infovqa',
    'kvqa',
    'gpt4v',
    'llavar',
    'scienceqa',
    'sharegpt4v',
    'stvqa',
    'super_clever',
    'textvqa',
    'tqa',
    'vsr',
    'icdar_2015',
    'lima',
    'alpaca-gpt4',
    'openhermes2.5',
    'mini-gemini',
    'hme100k',
    'tabwp_cot',
    'geo3k',
    'clevr_math_5w',
    'poie',
    'lvis_instruct4v_cap',
    'MetaMathQA',
    'MathInstruct',
    'orca-math-word-problems-200k',
    'math',
    '500k-atlas-math',
    'gpt4o-complex-20240809-en',
    'MathV360K'
]

if __name__ == '__main__':
    root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2'
    filenames_cur_rank = dist_split_files(points)
    world_size, rank, _ = get_distributed_env()
    success_data = []
    failed_data = []
    for filename in tqdm(filenames_cur_rank):
        print(f"Processing {filename}")
        file_root = os.path.join(
            root, filename, 'jsonl/qwen2vl-grammar-correct')
        output_root = os.path.join(
            root, filename, 'recordio/qwen2vl-grammar-correct-yi-1-5')
        os.makedirs(output_root, exist_ok=True)
        files = find_all_files(file_root, 'jsonl')
        files = [f for f in files if 'data_' not in f]
        if len(files) == 0:
            print(f"No files found in {file_root}")
            failed_data.append(filename)
            continue
        files_str = ' '.join(files)
        output_data = os.path.join(output_root, f'{filename}_data.recordio')
        image_data = os.path.join(output_root, f'{filename}_image.recordio')
        cmd = (
            'mmq_prepare_multi_modal_data '
            f'--output_file {output_data} '
            f'--output_image_file {image_data} '
            '--tokenizer.type huggingface '
            '--tokenizer.vocab_file /mnt/cephfs/bensenliu/wfs/weights/nlp/Yi-1.5-9B-Chat '
            f'--input_files {files_str} '
            "--role_mapping $'user=<|im_start|>user\n,assistant=<|im_start|>assistant\n,system=<|im_start|>system\n' "
            '--concurrency 200 '
            '--chunk_size 1024 '
            '--num_modal_tokens 144 '
            "--eot_token '<|im_end|>' "
            '--special_masked_token "<|endoftext|>" '
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
