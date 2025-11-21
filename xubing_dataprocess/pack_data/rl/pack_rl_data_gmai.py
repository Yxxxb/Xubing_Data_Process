from uu import Error
import pandas as pd
import os
import glob
import numpy as np
import re
from tqdm import tqdm
import datasets
import base64
from datakit.utils.files import dump_list_to_jsonl_file, find_all_files, read_jsonl_file, mem_efficient_read_jsonl_file, read_parquet_file
import sys
from datakit.utils.image import decode_base64_image_to_pil
from PIL import Image
import io

from datasets import Dataset, concatenate_datasets

def mem_efficient_read_tsv_file(tsv_file: str):
    with open(tsv_file, 'r') as f:
        for line in f:
            if line.strip():
                # 使用制表符分割行，并创建一个字典或其他结构
                yield line.strip().split('\t')

def ensure_list(x):
    if isinstance(x, dict):
        return [x]
    if isinstance(x, list):
        return x
    return [x]

def index_to_letter(index):
    return chr(ord('A') + index)

# pip install pyarrow
parquets_path = "/mnt/cephfs/xubingye/wfs/datasets/rl-category/GMAI-MMBench"
use_val = False

save_path = parquets_path + "/data_verl_parquet_check"
os.makedirs(save_path, exist_ok=True)
parquet_files = []

for file in glob.glob(os.path.join(parquets_path, "*.tsv")):
    parquet_files.append(file)

format_prompt = " You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."
image_placeholder = "<image>"

datasets_list = []
sample_num = 0
df_val_save = pd.DataFrame(columns=['images', 'data_source', 'prompt', 'ability', 'reward_model', 'extra_info'])
for i, path in enumerate(parquet_files):
    lst = []
    print(f"dealing {i} file: {path}")
    for row in mem_efficient_read_tsv_file(path):
        if row[0] == 'index' or len(row) != 14:
            continue
        if row[8] == row[2] or row[8] in row[2] or row[2] in row[8]:
            answer = 'A'
        elif row[8] == row[3] or row[8] in row[3] or row[3] in row[8]:
            answer = 'B'
        elif row[8] == row[4] or row[8] in row[4] or row[4] in row[8]:
            answer = 'C'
        elif row[8] == row[5] or row[8] in row[5] or row[5] in row[8]:
            answer = 'D'
        elif row[8] == row[6] or row[8] in row[6] or row[6] in row[8]:
            answer = 'E'
        else:
            continue
        question = row[1]
        option_len = 0
        for j in range(2, 7):
            if row[j] != None and row[j] != '':
                option_len += 1
        for j in range(option_len):
            question += f" {index_to_letter(j)}: {row[j+2]}."
        
        if not row[7]:
            continue
        try:
            bytes_image = base64.b64decode(row[7])
        except Exception:
            continue  

        new_row = {
            'images': [{'bytes': bytes_image}],
            'data_source': "hiyouga/geometry3k",
            'prompt': ensure_list({'content': image_placeholder + question + format_prompt, 'role': 'user'}),
            'ability': "puzzle",
            'reward_model': {'ground_truth': answer, 'style': 'rule'},
            'extra_info': {'thinking': None, 'problem': question, 'solution': answer}
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

    if use_val:
        val_size = int(len(df_save) * 0.05)
        val_indices = np.random.choice(len(df_save), size=val_size, replace=False)
        val_df = df_save.iloc[val_indices].copy()
        df_val_save = pd.concat([df_val_save, val_df], ignore_index=True)
        df_save = df_save.drop(val_indices).reset_index(drop=True)


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