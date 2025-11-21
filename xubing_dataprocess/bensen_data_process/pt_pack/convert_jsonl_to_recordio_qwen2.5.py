import os
from datakit.utils.distributed import (dist_split_files,
                                       get_distributed_env)
from datakit.utils.files import (find_all_files)


if __name__ == '__main__':
    root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category-video/QA/LLaVA-Video-178K-refine/jsonl/qwen2vl-grammar_correct_1fps_64maxframes_holistic_encoding'
    output_root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category-video/QA/LLaVA-Video-178K-refine/recordio/qwen2vl-grammar_correct_1fps_64maxframes_holistic_encoding-test-qwen-2-5'
    os.makedirs(output_root, exist_ok=True)
    all_jsonls = find_all_files(root, 'jsonl')
    jsonls_cur_rank = dist_split_files(all_jsonls)
    world_size, rank, _ = get_distributed_env()
    num_jsonls_per_shard = 400
    num_slice_cur_rank = len(jsonls_cur_rank) // num_jsonls_per_shard + 1
    for i in range(num_slice_cur_rank):
        cur_jsonls = jsonls_cur_rank[i*num_jsonls_per_shard:(i+1)*num_jsonls_per_shard]
        jsonl_str = ' '.join(cur_jsonls)
        output_data = os.path.join(output_root, f'{rank}_{i}_data.recordio')
        image_data = os.path.join(output_root, f'{rank}_{i}_image.recordio')
        cmd = (
            'mmq_prepare_multi_modal_data '
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
            '--add_eot false'
        )
        os.system(cmd)
    print('Finished!')
