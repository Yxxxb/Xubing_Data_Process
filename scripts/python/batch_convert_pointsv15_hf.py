import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

parser = argparse.ArgumentParser(description='Convert ckpt to hf format')
parser.add_argument('--exps', nargs='+', help='List of experiments to convert')
parser.add_argument(
    '--root',
    type=str,
    default=  # noqa
    '/mnt/wfs/mmnanjingwfssh/project_pr-nlp-large_pretrain/xubingye/weights/vlm',  # noqa
    help='Root directory of ckpts')
parser.add_argument(
    '--clip',
    type=str,
    default=  # noqa
    '/mnt/cephfs/bensenliu/wfs/weights/cv/qwen2vl-7b-vision'  # noqa
)
parser.add_argument('--save-llm',
                    action='store_true',
                    help='Whether to store the weights of llm')
parser.add_argument('--is-video',
                    action='store_true',
                    help='Whether or not the checkpoint is video model.')
parser.add_argument('--is-omni',
                    action='store_true',
                    help='Whether or not the checkpoint is omni model.')
parser.add_argument('--max-pixels',
                    type=int,
                    default=12845056,
                    help='Maximum number of pixels in the input image.')
parser.add_argument('--is-rl-base',
                    action='store_true',
                    help='Whether or not the checkpoint is the base model for rl training.')

if __name__ == '__main__':
    args = parser.parse_args()
    for exp in tqdm(args.exps):
        ori_path = f'{args.root}/mmq-pointsv15-{exp}-sft'
        latest_txt = f'{ori_path}/latest_ckpt.txt'
        with open(latest_txt, 'r') as f:
            ckpt_name = f.read().strip()
        ori_path_iter = f'{ori_path}/{ckpt_name}'
        meta_path = f'{ori_path_iter}/meta.yaml'
        output_path = f'{ori_path}-hf' if not args.is_rl_base else f'{ori_path}-rl-base-hf'
        os.makedirs(output_path, exist_ok=True)
        cmd = ('welm_pointsv15_mmq_to_hf '
               f'--checkpoint_dir {ori_path_iter} '
               f'--output_dir {output_path} '
               f'--config_file {meta_path} '
               f'--llava.clip {args.clip}')
        if args.is_rl_base:
            cmd += ' --is_rl_base'
        print(cmd)
        os.system(cmd)
        with open(os.path.join(output_path, 'config.json'), 'r') as f:
            data = json.load(f)
            data['image_token_id'] = 151655
        with open(os.path.join(output_path, 'config.json'), 'w') as f:
            json.dump(data, f, indent=4)
        with open(os.path.join(output_path, 'preprocessor_config.json'),
                  'r') as f:
            data = json.load(f)
            data['processor_class'] = 'Qwen2VLProcessor'
            data['max_pixels'] = args.max_pixels
            data['size']['max_pixels'] = args.max_pixels
        with open(os.path.join(output_path, 'preprocessor_config.json'),
                  'w') as f:
            json.dump(data, f, indent=4)
        if args.save_llm:
            assert not args.is_rl_base, "save_llm and is_video should be set to False when converting rl weights"
            llm_output_path = output_path.replace('-hf', '-llm-hf')
            model = AutoModelForCausalLM.from_pretrained(
                output_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map='cuda')
            model.llm.save_pretrained(llm_output_path,
                                      max_shard_size='5GB',
                                      safe_serialization=True)
            os.system(f'rm -rf {llm_output_path}/*.json')
            if args.is_omni:
                ori_json = '/mnt/cephfs/bensenliu/wfs/weights/mm/mmq-pointsv15-20250328e1-sft-llm-hf/*.json'  # noqa
                ori_txt = '/mnt/cephfs/bensenliu/wfs/weights/mm/mmq-pointsv15-20250328e1-sft-llm-hf/*.txt'  # noqa
            else:
                ori_json = '/mnt/cephfs/bensenliu/wfs/weights/nlp/Qwen2.5-7B-Instruct/*.json'  # noqa
                ori_txt = '/mnt/cephfs/bensenliu/wfs/weights/nlp/Qwen2.5-7B-Instruct/*.txt'  # noqa
            os.system(f'cp {ori_json} {llm_output_path}'  # noqa
                      )
            os.system(f'cp {ori_txt} {llm_output_path}'  # noqa
                      )
        if args.is_video:
            assert not args.is_rl_base, "save_llm and is_video should be set to False when converting rl weights"
            video_output_path = output_path.replace('-hf', '-video-hf')
            os.system(f'cp -r {output_path} {video_output_path}')
            os.system(
                f'cp /mnt/cephfs/bensenliu/exp_runs/weights/mm/mmq-pointsv15-20250225e1-sft-hf/preprocessor_config.json {video_output_path}'  # noqa
            )  # noqa
        if args.is_rl_base:
            with open(os.path.join(output_path, 'generation_config.json'), 'r') as f:
                data = json.load(f)
                data['bos_token_id'] = 151643
                data['pad_token_id'] = 151643
                data['eos_token_id'] = [151645, 151643]
            with open(os.path.join(output_path, 'generation_config.json'), 'w') as f:
                json.dump(data, f, indent=4)
            with open(os.path.join(output_path, 'preprocessor_config.json'),
                    'r') as f:
                data = json.load(f)
                if "size" in data:
                    del data['size']
            with open(os.path.join(output_path, 'preprocessor_config.json'), 'w') as f:
                json.dump(data, f, indent=4)
