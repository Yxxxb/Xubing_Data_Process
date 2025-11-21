import pandas as pd
import os
import glob
import numpy as np
import re
from tqdm import tqdm
import datasets

def ensure_list(x):
    if isinstance(x, dict):
        return [x]
    if isinstance(x, list):
        return x
    return [x]

def index_to_letter(index):
    return chr(ord('A') + index)

def extract_list_from_str(answer):
    match = re.match(r"^\['(.*?)'\]$", answer)
    if match:
        return match.group(1)
    else:
        return answer

df_example = pd.read_parquet("/mnt/cephfs/xubingye/wfs/datasets/rl-category/GEOQA_R1V_Train_8K/data_verl_parquet/train-00000-of-00001.parquet")

# pip install pyarrow
parquets_path = "/mnt/cephfs/xubingye/wfs/datasets/rl-category/SFE/20250611-en"
use_val = False

save_path = parquets_path + "_verl_parquet"
os.makedirs(save_path, exist_ok=True)
parquet_files = []
for file in glob.glob(os.path.join(parquets_path, "*.parquet")):
    df = pd.read_parquet(file)
    parquet_files.append(df)

format_prompt = " You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."
image_placeholder = "<image>"

lst = []
drop_num = 0
sample_num = 0
df_val_save = pd.DataFrame(columns=['images', 'data_source', 'prompt', 'ability', 'reward_model', 'extra_info'])
for i, df in enumerate(parquet_files):
    print(f"Processing file {i + 1}/{len(parquet_files)}: {df.shape}")
    # breakpoint()
    df_save = pd.DataFrame(columns=['images', 'data_source', 'prompt', 'ability', 'reward_model', 'extra_info'])
    for index, row in tqdm(df.iterrows()):
        if len(row['images']) != 1:
            images = []
            for _image in row['images']:
                img_path = "/mnt/cephfs/xubingye/wfs/datasets/rl-category/SFE/images/" + _image.split('/')[-1]
                with open(img_path, 'rb') as f:
                    img_bytes = f.read()
                images.append({'bytes': img_bytes, 'path': img_path})
            problem = row['question']
            for option in row['options']:
                problem += f" {option}"
            if isinstance(row['answer'], list):
                assert len(row['answer']) == 1, print(row['answer'])
                answer = row['answer'][0]
            else:
                answer = extract_list_from_str(row['answer'])
        else:
            img_path = "/mnt/cephfs/xubingye/wfs/datasets/rl-category/SFE/images/" + row['images'][0].split('/')[-1]
            with open(img_path, 'rb') as f:
                img_bytes = f.read()
            images = {'bytes': img_bytes, 'path': img_path}
            problem = image_placeholder + row['question'].replace(image_placeholder, '')
            for option in row['options']:
                problem += f" {option}"
            if isinstance(row['answer'], list):
                assert len(row['answer']) == 1, print(row['answer'])
                answer = row['answer'][0]
            else:
                answer = extract_list_from_str(row['answer'])
        new_row = {
            'images': ensure_list(images),
            'data_source': "hiyouga/geometry3k",
            'prompt': ensure_list({'content': problem + format_prompt, 'role': 'user'}),
            'ability': "puzzle",
            'reward_model': {'ground_truth': answer, 'style': 'rule'},
            'extra_info': {'thinking': row.get('pattern', ''), 'problem': row['question'], 'solution': answer}
        }
        df_save = pd.concat([df_save, pd.DataFrame([new_row])], ignore_index=True)
        sample_num += 1
    # 从df_save中随机抽出百分之五作为valset
    if use_val:
        val_size = int(len(df_save) * 0.05)
        val_indices = np.random.choice(len(df_save), size=val_size, replace=False)
        val_df = df_save.iloc[val_indices].copy()
        df_val_save = pd.concat([df_val_save, val_df], ignore_index=True)
        df_save = df_save.drop(val_indices).reset_index(drop=True)

    # 如果需要保存修改后的DataFrame
    total_files = len(parquet_files)
    filename = f"train-{i:05d}-of-{total_files:05d}.parquet"
    df_save.to_parquet(os.path.join(save_path, filename))

# 保存valset
if use_val:
    val_filename = f"validation.parquet"
    val_save_path = save_path + "_val"
    os.makedirs(val_save_path, exist_ok=True)
    df_val_save.to_parquet(os.path.join(val_save_path, val_filename))

print(sample_num)