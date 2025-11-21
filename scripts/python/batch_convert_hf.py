import argparse
import os

from tqdm import tqdm

parser = argparse.ArgumentParser(description='Convert ckpt to hf format')
parser.add_argument('--exps', nargs='+', help='List of experiments to convert')
parser.add_argument(
    '--root',
    type=str,
    default=  # noqa
    '/mnt/wfs/mmnanjingwfssh/project_pr-nlp-large_pretrain/bensenliu/weights/mm',  # noqa
    help='Root directory of ckpts')
parser.add_argument(
    '--clip',
    type=str,
    default=  # noqa
    '/mnt/cephfs/bensenliu/exp_runs/weights/mm/CLIP-L-336/clip-vit-large-patch14-336'  # noqa
)

if __name__ == '__main__':
    args = parser.parse_args()
    for exp in tqdm(args.exps):
        ori_path = f'{args.root}/mmq-llava-{exp}-sft'
        latest_txt = f'{ori_path}/latest_ckpt.txt'
        with open(latest_txt, 'r') as f:
            ckpt_name = f.read().strip()
        ori_path_iter = f'{ori_path}/{ckpt_name}'
        meta_path = f'{ori_path_iter}/meta.yaml'
        output_path = f'{ori_path}-hf'
        os.makedirs(output_path, exist_ok=True)
        cmd = ('welm_llava_mmq_to_hf '
               f'--checkpoint_dir {ori_path_iter} '
               f'--output_dir {output_path} '
               f'--config_file {meta_path} '
               f'--llava.clip {args.clip}')
        print(cmd)
        os.system(cmd)
