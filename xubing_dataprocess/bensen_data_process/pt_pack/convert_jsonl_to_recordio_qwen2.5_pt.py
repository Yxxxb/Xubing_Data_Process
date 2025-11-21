from datakit.utils.files import find_all_files
from datakit.utils.distributed import dist_split_files, get_distributed_env
import os

if __name__ == '__main__':
    jsonl_folder = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/pt/obelics/jsonl/qwen2vl-grammar_correct_1M'
    output_folder = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/pt/obelics/recordio/qwen2vl-grammar_correct_1M-qwen-2-5'

    os.makedirs(output_folder, exist_ok=True)

    num_jsonls_per_shard = 400
    jsonls = find_all_files(jsonl_folder, 'jsonl')
    jsonls_cur_rank = dist_split_files(jsonls)
    world_size, rank, local_rank = get_distributed_env()
    num_slice_cur_rank = len(jsonls_cur_rank) // num_jsonls_per_shard + 1
    for i in range(num_slice_cur_rank):
        cur_jsonls = jsonls_cur_rank[i *
                                     num_jsonls_per_shard: (i + 1) * num_jsonls_per_shard]
        json_files_str = ' '.join(cur_jsonls)
        output_data = os.path.join(output_folder, f'{rank}_{i}_data.recordio')
        image_data = os.path.join(output_folder, f'{rank}_{i}_image.recordio')

        cmd = (
            'mmq_prepare_multi_modal_data '
            f'--output_file {output_data} '
            f'--output_image_file {image_data} '
            '--tokenizer.type huggingface '
            '--tokenizer.vocab_file /mnt/cephfs/bensenliu/wfs/weights/nlp/Qwen2.5-7B-Instruct '
            f'--input_files {json_files_str} '
            '--concurrency 200 '
            '--chunk_size 1024 '
            '--num_modal_tokens 144 '
            "--eot_token '<|im_end|>' "
            '--special_masked_token "<|image_pad|>" '
            '--no_role '
            '--add_eot false'
        )
        os.system(cmd)

    print('====> Done <=====')
