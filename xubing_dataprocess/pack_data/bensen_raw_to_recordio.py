import argparse
import os
import subprocess

from datakit.utils.distributed import (barrier_all_processes, delete_folder,
                                       dist_split_files, get_distributed_env)
from datakit.utils.files import find_all_files

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input-folder',
    type=str,
    required=True,  # noqa
    help='The folder containing jsonl files to be converted to recordio')
parser.add_argument('--output-folder',
                    type=str,
                    required=True,
                    help='The folder to store the converted recordio files')
parser.add_argument('--task-name',
                    type=str,
                    required=True,
                    help='The name of this task, used for synchronization')
parser.add_argument('--batch-size', type=int, default=512, help='Batch size')
parser.add_argument('--is-sft', action='store_true', help='Whether to use SFT')
parser.add_argument(
    '--model-path',
    type=str,
    default=  # noqa
    '/mnt/wfs/mmnanjingwfssh/project_pr-nlp-large_pretrain/bensenliu/weights/nlp/Qwen2.5-7B-Instruct'  # noqa
)

if __name__ == '__main__':
    args = parser.parse_args()
    delete_folder(args.task_name, is_cpu=True)
    input_folder = args.input_folder
    output_folder = args.output_folder
    batch_size = args.batch_size
    os.makedirs(output_folder, exist_ok=True)
    all_jsonls = find_all_files(input_folder, 'jsonl')
    jsonls_cur_rank = dist_split_files(all_jsonls, is_cpu=True)
    _, rank, _ = get_distributed_env(is_cpu=True)
    print(f'Rank {rank} processing {len(jsonls_cur_rank)} jsonls')
    for i in range(0, len(jsonls_cur_rank), batch_size):
        cur_batch_jsonls = jsonls_cur_rank[i:i + batch_size]
        output_data_recordio = os.path.join(
            output_folder, f'{rank}_{i//batch_size}_data.recordio')
        output_image_recordio = os.path.join(
            output_folder, f'{rank}_{i//batch_size}_image.recordio')
        if args.is_sft:
            cmd = (
                'mmq_prepare_multi_modal_data '
                f'--output_file {output_data_recordio} '
                f'--output_image_file {output_image_recordio} '
                '--tokenizer.type huggingface '
                f'--tokenizer.vocab_file {args.model_path} '  # noqa
                f'--input_files {" ".join(cur_batch_jsonls)} '
                "--role_mapping 'user=<|im_start|>user\\n,assistant=<|im_start|>assistant\\n,system=<|im_start|>system\\n' "  # noqa
                '--concurrency 200 '
                '--chunk_size 1024 '
                '--num_modal_tokens 144 '
                "--eot_token '<|im_end|>' "
                '--special_masked_token "<|image_pad|>" '
                '--mask_user '
                '--add_eot false')
        else:
            cmd = ('mmq_prepare_multi_modal_data '
                   f'--output_file {output_data_recordio} '
                   f'--output_image_file {output_image_recordio} '
                   '--tokenizer.type huggingface '
                   f'--tokenizer.vocab_file {args.model_path} '
                   f'--input_files {" ".join(cur_batch_jsonls)} '
                   '--concurrency 300 '
                   '--chunk_size 1024 '
                   '--num_modal_tokens 144 '
                   "--eot_token '<|im_end|>' "
                   '--special_masked_token "<|image_pad|>" '
                   '--add_eot false '
                   '--no_role')
        print('=============== Command ===================\n')
        print(cmd)
        print('\n=========================================\n')
        result = subprocess.run(cmd,
                                shell=True,
                                text=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        if result.returncode == 0:
            print(
                f'Successfully converted jsonl to recordio: {output_data_recordio}'  # noqa
            )
        else:
            raise ValueError(
                f'Failed to convert jsonl to recordio: {result.stdout}')
    barrier_all_processes(args.task_name, is_cpu=True)
