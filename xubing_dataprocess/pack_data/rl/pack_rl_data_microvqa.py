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


# df_path = "/mnt/cephfs/xubingye/wfs/datasets/rl-category/microvqa/data_verl_parquet"
# dfs = find_all_files(df_path, "parquet")
# # 初始化一个空的DataFrame用于存储拼接后的数据
# combined_df = pd.DataFrame(columns=['images', 'data_source', 'prompt', 'ability', 'reward_model', 'extra_info'])
# max_len = 0
# for parquet_file in tqdm(dfs, desc="Reading parquet files"):
#     df = pd.read_parquet(parquet_file)
#     if 'extra_info' in df.columns:
#         df = df.drop(columns=['extra_info'])
#         df = df.drop(columns=['images'])
#     combined_df = pd.concat([combined_df, df], ignore_index=True)
# combined_df.to_parquet('/mnt/cephfs/xubingye/wfs/datasets/rl-category/microvqa/data_verl_parquet_check/train.parquet')
# breakpoint()

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
parquets_path = "/mnt/cephfs/xubingye/wfs/datasets/rl-category/microvqa/data"
use_val = False

save_path = parquets_path + "_verl_parquet_check"
os.makedirs(save_path, exist_ok=True)
parquet_files = []
for file in glob.glob(os.path.join(parquets_path, "*.parquet")):
    df = pd.read_parquet(file)
    parquet_files.append(df)

format_prompt = " You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."
image_placeholder = "<image>"



datasets_list = []
multi_img_num = 0
sample_num = 0
df_val_save = pd.DataFrame(columns=['images', 'data_source', 'prompt', 'ability', 'reward_model', 'extra_info'])
for i, df in enumerate(parquet_files):
    lst = []
    print(f"Processing file {i + 1}/{len(parquet_files)}: {df.shape}")
    df_save = pd.DataFrame(columns=['images', 'data_source', 'prompt', 'ability', 'reward_model', 'extra_info'])
    for index, row in tqdm(df.iterrows()):
        answer = index_to_letter(row['correct_index'])
        question = row['question']
        for idx, choice in enumerate(row['choices']):
            question += f" {index_to_letter(idx)}: {choice}."
        # if len(row['images_list']) > 1:
        #     multi_img_num += 1
        #     continue
        new_row = {
            'images': row['images_list'],
            'data_source': "hiyouga/geometry3k",
            'prompt': ensure_list({'content': image_placeholder + question + format_prompt, 'role': 'user'}),
            'ability': "puzzle",
            'reward_model': {'ground_truth': answer, 'style': 'rule'},
            'extra_info': {'thinking': row.get('pattern', ''), 'problem': question, 'solution': answer}
        }
        lst.append(new_row)
        sample_num += 1
    
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