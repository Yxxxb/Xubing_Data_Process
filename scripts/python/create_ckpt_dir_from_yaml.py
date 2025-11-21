import argparse
import os

import yaml

parser = argparse.ArgumentParser(
    'Checkpoint directory creation from yaml config')
parser.add_argument('--config', type=str, required=True)
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)


def recursive_traversal(conf: dict):
    if isinstance(conf, dict):
        for k, v in conf.items():
            if isinstance(v, dict):
                recursive_traversal(v)
            elif isinstance(k, str) and k in ('save_dir', 'save_path'):
                ckpt_dir = v
                print(f'CREATE {k} AS {v}')
                os.makedirs(ckpt_dir, exist_ok=True)


recursive_traversal(config)
