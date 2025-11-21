import argparse
import os
from typing import Callable, List, Optional

import yaml

from .distributed import (barrier_all_processes, get_distributed_env,
                          gpu_utilization, kill_process)
from .files import find_all_files, read_mmq_index
from .utils import run_command_and_get_return_code


def generate_index(root_list: str,
                   index_file_root: str,
                   index_file_name: str,
                   base_train_config: str,
                   train_config_file: str,
                   model_save_folder: str,
                   alert_path: str,
                   base_datasets_file: str = None,
                   num_epochs: int = 1,
                   seq_length: int = 32768,
                   batch_size: int = 64) -> str:
    """
    Generate index and training configuration.

    Args:
        base_datasets_file (str): Path to the base datasets file.
        root_list (str): Root directory for the datasets, separated by comma.
        index_file_root (str): Root folder to save the index file.
        index_file_name (str): Name of the saved index file, e.g., 202503/20250327/e1_sft_index.recordio.
        num_epochs (int, optional): Number of epochs for training. Default is 1.
        seq_length (int, optional): Sequence length for training. Default is 8192.
        batch_size (int, optional): Batch size for training. Default is 64.
        base_train_config (str): Path to the base training configuration file. e.g., '/mnt/cephfs/bensenliu/code/mimikyu_h800/configs/202503/20250325/20250325e2-sft.yaml'
        train_config_file (str): Path to the training configuration file.
        model_save_folder (str): Folder to save model weights. e.g., '/mnt/cephfs/bensenliu/wfs/weights/mm/'

    Returns:
        None
    """
    gpu_utilization()
    world_size, rank, _ = get_distributed_env()
    if rank != 0:
        barrier_all_processes(task_name='recordio_index', root=alert_path)
        kill_process()
        return
    base_datasets = []
    if base_datasets_file is None:
        with open(base_train_config, 'r') as f:
            base_train_config_file = yaml.load(f, Loader=yaml.FullLoader)
            base_datasets_file = base_train_config_file['data']['train'][
                'index_file']
            reader = read_mmq_index(base_datasets_file)
            base_datasets = reader.header.filenames
    reader = read_mmq_index(base_datasets_file)
    base_datasets = reader.header.filenames
    new_datasets = []
    root_list = root_list.split(',')
    for root in root_list:
        root = root.strip()
        new_dataset = find_all_files(root, 'data.recordio')
        new_datasets.extend(new_dataset)
    datasets = base_datasets + new_datasets
    index_file = os.path.join(index_file_root, index_file_name)
    os.makedirs(os.path.dirname(index_file), exist_ok=True)
    cmd = ['mmq_batch_collator', '--input_files']
    cmd.extend(datasets)
    cmd.extend([
        '--output_file',
        index_file,
        '--num_epochs',
        str(num_epochs),
        '--batch_strategy',
        'by_token',
        '--micro_batch_tokens_limit',
        str(seq_length),
        '--batch_tokens',
        str(seq_length * batch_size),
        '--split_doc',
        'False',
    ])
    print(' '.join(cmd))
    print('Please wait for the index file to be generated...')
    return_code, _, stderr = run_command_and_get_return_code(cmd)
    if return_code != 0:
        raise ValueError(f'Failed to generate index file: {stderr}')
    print(f'Successfully generate index file: {index_file}')

    iteration = int(stderr.strip().split('\n')[-1].split('it')[0])
    with open(base_train_config, 'r') as f:
        base_train_config = yaml.load(f, Loader=yaml.FullLoader)
    base_train_config['optimizer']['warmup_iters'] = int(iteration * 0.03)
    base_train_config['optimizer']['lr_decay_iters'] = iteration - int(
        iteration * 0.03)
    base_train_config['data']['train']['index_file'] = index_file
    exp_index = index_file_name.split('/')[-1].split('_')[0]
    base_train_config['checkpoint']['save_dir'] = os.path.join(
        model_save_folder,
        f'mmq-pointsv15-{index_file_name.split("/")[-2]}{exp_index}-sft'
    )  # noqa
    base_train_config['log']['wandb'][
        'group'] = f"pointsv15_{index_file_name.split(' / ')[-2]}{exp_index}_sft"  # noqa
    os.makedirs(os.path.dirname(train_config_file), exist_ok=True)
    with open(train_config_file, 'w') as f:
        yaml.dump(base_train_config, f)

    print(f'Write config file to {train_config_file}')
    barrier_all_processes(task_name='recordio_index', root=alert_path)
    kill_process()
    print(f'Index file: {index_file}')
