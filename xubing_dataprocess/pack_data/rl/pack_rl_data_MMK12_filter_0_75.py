import pandas as pd
import os
import glob
import numpy as np
import re
from tqdm import tqdm
import datasets
from typing import List, Optional, Tuple
from datakit.utils.files import dump_list_to_jsonl_file, find_all_files, read_jsonl_file, mem_efficient_read_jsonl_file, read_parquet_file
import base64


# df = pd.read_parquet("/mnt/cephfs/xubingye/wfs/datasets/rl-category/MMK12/data_verl_parquet/train-00000-of-00004.parquet")
# js = read_jsonl_file("/mnt/cephfs/xubingye/wfs/datasets/rl-category/MMK12/data_verl_parquet_offline_roll_filter_result_exactly_match/0/train-00003-of-00004.jsonl")
# breakpoint()

def ensure_list(x):
    if isinstance(x, dict):
        return [x]
    if isinstance(x, list):
        return x
    return [x]

# pip install pyarrow
parquets_path = "/mnt/cephfs/xubingye/wfs/datasets/rl-category/MMK12/data_verl_parquet_offline_roll_filter_result_exactly_match"
use_val = False

save_path = "/mnt/cephfs/xubingye/wfs/datasets/rl-category/MMK12/data_verl_parquet_filter_0_75_offline_roll"
os.makedirs(save_path, exist_ok=True)
parquet_files = []
parquet_paths = find_all_files(parquets_path, 'jsonl')
for parquet_path in parquet_paths:
    df = read_jsonl_file(parquet_path)
    parquet_files.append(df)

df_val_save = pd.DataFrame(columns=['images', 'data_source', 'prompt', 'ability', 'reward_model', 'extra_info'])
df_save = pd.DataFrame(columns=['images', 'data_source', 'prompt', 'ability', 'reward_model', 'extra_info'])
for i, df in enumerate(parquet_files):
    print(f"Processing file {i + 1}/{len(parquet_files)}: {len(df)}")
    for row in tqdm(df):
        if row['match_num'] <= 0 or row['match_num'] >= 13:
            continue
        new_row = {
            'images': ensure_list({'bytes': base64.b64decode(row['images']), 'path': row['image_path_id']}),
            'data_source': "FanqingM/MMK12",
            'prompt': ensure_list(row['prompt']),
            'ability': "Math",
            'reward_model': row['reward_model'],
            'extra_info': row['extra_info']
        }
        df_save = pd.concat([df_save, pd.DataFrame([new_row])], ignore_index=True)
    # 从df_save中随机抽出百分之五作为valset
    if use_val:
        val_size = int(len(df_save) * 0.03)
        val_indices = np.random.choice(len(df_save), size=val_size, replace=False)
        val_df = df_save.iloc[val_indices].copy()
        df_val_save = pd.concat([df_val_save, val_df], ignore_index=True)
        df_save = df_save.drop(val_indices).reset_index(drop=True)

# 如果需要保存修改后的DataFrame
filename = f"train-{0:05d}-of-{0:05d}.parquet"
df_save.to_parquet(os.path.join(save_path, filename))

# 保存valset
if use_val:
    val_filename = f"validation.parquet"
    val_save_path = save_path + "_val"
    os.makedirs(val_save_path, exist_ok=True)
    df_val_save.to_parquet(os.path.join(val_save_path, val_filename))
