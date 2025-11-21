import pandas as pd
import os
import glob
import numpy as np
import re
from tqdm import tqdm
import datasets
from datakit.utils.files import read_jsonl_file

from datasets import Dataset, concatenate_datasets

def ensure_list(x):
    if isinstance(x, dict):
        return [x]
    if isinstance(x, list):
        return x
    return [x]

def index_to_letter(index):
    return chr(ord('A') + index)

df_example = pd.read_parquet("/mnt/cephfs/xubingye/wfs/datasets/rl-category/GEOQA_R1V_Train_8K/data_verl_parquet/train-00000-of-00001.parquet")

# pip install pyarrow
parquets_path = "/mnt/cephfs/xubingye/wfs/datasets/rl-category/VisuLogic"
use_val = False

save_path = parquets_path + "/data_verl_parquet_check"
os.makedirs(save_path, exist_ok=True)
parquet_files = read_jsonl_file(parquets_path + "/data.jsonl")
parquet_files = [parquet_files]

format_prompt = " You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."
image_placeholder = "<image>"

datasets_list = []
lst = []

sample_num = 0
df_val_save = pd.DataFrame(columns=['images', 'data_source', 'prompt', 'ability', 'reward_model', 'extra_info'])
for i, df in enumerate(parquet_files):
    print(f"Processing file {i + 1}/{len(parquet_files)}")
    df_save = pd.DataFrame(columns=['images', 'data_source', 'prompt', 'ability', 'reward_model', 'extra_info'])
    for index, row in enumerate(df):
        image_path = "/mnt/cephfs/xubingye/wfs/datasets/rl-category/VisuLogic/" + row['image_path']
        with open(image_path, 'rb') as f:
            img_bytes = f.read()
        new_row = {
            'images': ensure_list({'bytes': img_bytes, 'path': image_path}),
            'data_source': "hiyouga/geometry3k",
            'prompt': ensure_list({'content': image_placeholder + row['question'] + format_prompt, 'role': 'user'}),
            'ability': "puzzle",
            'reward_model': {'ground_truth': row['label'], 'style': 'rule'},
            'extra_info': {'thinking': row.get('pattern', ''), 'problem': row['question'], 'solution': row['label']}
        }
        lst.append(new_row)
        sample_num += 1

        if len(lst) > 20:
            sub_dataset = Dataset.from_list(lst)
            datasets_list.append(sub_dataset)
            lst = []

        # breakpoint()
    # 从df_save中随机抽出百分之五作为valset
    if use_val:
        val_size = int(len(df_save) * 0.05)
        val_indices = np.random.choice(len(df_save), size=val_size, replace=False)
        val_df = df_save.iloc[val_indices].copy()
        df_val_save = pd.concat([df_val_save, val_df], ignore_index=True)
        df_save = df_save.drop(val_indices).reset_index(drop=True)

    # 如果需要保存修改后的DataFrame
    # total_files = len(parquet_files)
    # filename = f"train-{i:05d}-of-{total_files:05d}.parquet"
    # df_save.to_parquet(os.path.join(save_path, filename))

if len(lst) > 0:
    sub_dataset = Dataset.from_list(lst)
    datasets_list.append(sub_dataset)
    lst = []

dataset = concatenate_datasets(datasets_list)
save_path_parquet = os.path.join(save_path, "train-sum.parquet")
dataset.to_parquet(save_path_parquet)

# 保存valset
if use_val:
    val_filename = f"validation.parquet"
    val_save_path = save_path + "_val"
    os.makedirs(val_save_path, exist_ok=True)
    df_val_save.to_parquet(os.path.join(val_save_path, val_filename))

print(sample_num)