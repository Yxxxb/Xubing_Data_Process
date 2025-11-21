import pandas as pd
import os
import glob
import numpy as np
import re
from tqdm import tqdm
import datasets
from datakit.utils.files import dump_list_to_jsonl_file, find_all_files, read_jsonl_file, mem_efficient_read_jsonl_file, read_parquet_file
import sys
from datakit.utils.image import decode_base64_image_to_pil, encode_np_to_base64_image, encode_bytes_to_base64_image
from PIL import Image
import io

from datasets import Dataset, concatenate_datasets

def ensure_list(x):
    if isinstance(x, dict):
        return [x]
    if isinstance(x, list):
        return x
    return [x]

def index_to_letter(index):
    return chr(ord('A') + index)

# val = pd.read_parquet("/mnt/cephfs/xubingye/wfs/datasets/rl-category/CharXiv/val.parquet")
# test = pd.read_parquet("/mnt/cephfs/xubingye/wfs/datasets/rl-category/CharXiv/test.parquet")
# breakpoint()

# pip install pyarrow
parquets_path = "/mnt/cephfs/xubingye/wfs/datasets/rl-category/CharXiv"
use_val = False

save_path = parquets_path + "/data_verl_parquet_check"
os.makedirs(save_path, exist_ok=True)
parquet_files = []
for file in glob.glob(os.path.join(parquets_path, "*.parquet")):
    if 'val.parquet' not in file:
        continue
    df = pd.read_parquet(file)
    parquet_files.append(df)

format_prompt = " You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."
image_placeholder = "<image>"

breakpoint()

datasets_list = []
multi_img_num = 0
sample_num = 0
df_val_save = pd.DataFrame(columns=['images', 'data_source', 'prompt', 'ability', 'reward_model', 'extra_info'])
for i, df in enumerate(parquet_files):
    lst = []
    print(f"Processing file {i + 1}/{len(parquet_files)}: {df.shape}")
    df_save = pd.DataFrame(columns=['images', 'data_source', 'prompt', 'ability', 'reward_model', 'extra_info'])
    for index, row in tqdm(df.iterrows()):
        if index == 0:
            continue
        answer = row['reasoning_a']
        question = row['reasoning_q']
        new_row = {
            'images': ensure_list(row['image']),
            'data_source': "hiyouga/geometry3k",
            'prompt': ensure_list({'content': image_placeholder + question + format_prompt, 'role': 'user'}),
            'ability': "puzzle",
            'reward_model': {'ground_truth': answer, 'style': 'rule'},
            'extra_info': {'thinking': row.get('pattern', ''), 'problem': question, 'solution': answer}
        }
        lst.append(new_row)
        sample_num += 1

        if len(lst) > 20:
            sub_dataset = Dataset.from_list(lst)
            datasets_list.append(sub_dataset)
            lst = []
    
    if len(lst) > 0:
        sub_dataset = Dataset.from_list(lst)
        datasets_list.append(sub_dataset)

    # 从df_save中随机抽出百分之五作为valset
    if use_val:
        val_size = int(len(df_save) * 0.05)
        val_indices = np.random.choice(len(df_save), size=val_size, replace=False)
        val_df = df_save.iloc[val_indices].copy()
        df_val_save = pd.concat([df_val_save, val_df], ignore_index=True)
        df_save = df_save.drop(val_indices).reset_index(drop=True)



# 保存valset
if use_val:
    val_filename = f"validation.parquet"
    val_save_path = save_path + "_val"
    os.makedirs(val_save_path, exist_ok=True)
    df_val_save.to_parquet(os.path.join(val_save_path, val_filename))

dataset = concatenate_datasets(datasets_list)
save_path_parquet = os.path.join(save_path, "train-sum.parquet")
dataset.to_parquet(save_path_parquet)