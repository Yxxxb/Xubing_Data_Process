import pandas as pd
import os
import glob
import numpy as np
import re
from tqdm import tqdm
import datasets

from datasets import Dataset, concatenate_datasets

# data_path = "/mnt/cephfs/xubingye/wfs/datasets/rl-category/Clevr_CoGenT_TrainA_R1/data/train-00000-of-00015.parquet"
# tfile_path = '/mnt/cephfs/xubingye/wfs/datasets/rl-category/geometry3k_proc/train.parquet'
# vfile_path = '/mnt/cephfs/xubingye/wfs/datasets/rl-category/geometry3k_proc/test.parquet'
# new_path = "/mnt/cephfs/xubingye/wfs/datasets/rl-category/Clevr_CoGenT_TrainA_R1/data_verl_parquet/train-00000-of-00015.parquet"
# tdf = pd.read_parquet(tfile_path)
# vdf = pd.read_parquet(vfile_path)
# ndf = pd.read_parquet(new_path)
# data_df = pd.read_parquet(data_path)

# dataframet = datasets.load_dataset("parquet", data_files=tfile_path)["train"]
# dataframen = datasets.load_dataset("parquet", data_files=new_path)["train"]
# # print("###############", dataframe.iloc[0])

# breakpoint()

def ensure_list(x):
    if isinstance(x, dict):
        return [x]
    if isinstance(x, list):
        return x
    return [x]

def index_to_letter(index):
    return chr(ord('A') + index)

df_example1 = pd.read_parquet("/mnt/cephfs/xubingye/wfs/datasets/rl-category/journeybench-multi-image-vqa/data/train-00000-of-00001.parquet")
df_example2 = pd.read_parquet("/mnt/cephfs/xubingye/wfs/datasets/rl-category/journeybench-multi-image-vqa/data_verl_parquet/train-00000-of-00002.parquet")

# pip install pyarrow
parquets_path = "/mnt/cephfs/xubingye/wfs/datasets/rl-category/EMMA/data"
use_val = False

save_path = parquets_path + "_verl_parquet_check"
os.makedirs(save_path, exist_ok=True)

parquets_paths = [
    "/mnt/cephfs/xubingye/wfs/datasets/rl-category/EMMA/Chemistry/",
    "/mnt/cephfs/xubingye/wfs/datasets/rl-category/EMMA/Coding/",
    "/mnt/cephfs/xubingye/wfs/datasets/rl-category/EMMA/Math/",
    "/mnt/cephfs/xubingye/wfs/datasets/rl-category/EMMA/Physics/",
]
parquet_files = []
for parquets_path in parquets_paths:
    for file in glob.glob(os.path.join(parquets_path, "*.parquet")):
        df = pd.read_parquet(file)
        parquet_files.append(df)

format_prompt = " You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."
image_placeholder = "<image>"

datasets_list = []
lst = []

sample_num = 0
df_val_save = pd.DataFrame(columns=['images', 'data_source', 'prompt', 'ability', 'reward_model', 'extra_info'])
for i, df in enumerate(parquet_files):
    print(f"Processing file {i + 1}/{len(parquet_files)}: {df.shape}")
    df_save = pd.DataFrame(columns=['images', 'data_source', 'prompt', 'ability', 'reward_model', 'extra_info'])
    for index, row in tqdm(df.iterrows()):
        # breakpoint()
        assert 'image_5' in row and 'image_6' not in row
        images = [image for image in [row['image_1'], row['image_2'], row['image_3'], row['image_4'], row['image_5']] if image is not None]
        if row['options'] is not None and len(row['options']) > 0:
            problem = row['question']
            for q_i in range(len(row['options'])):
                problem += f" {index_to_letter(q_i)}: {row['options'][q_i]}."
        else:
            problem = row['question']
            
        seen_placeholders = set()
        new_problem = []
        words = re.split(r'(\<image_\d+\>)', problem)
        for word in words:
            if word.startswith('<image_') and word.endswith('>'):
                if word not in seen_placeholders:
                    seen_placeholders.add(word)
                    new_problem.append(word)
            else:
                new_problem.append(word)
        problem = ''.join(new_problem)

        assert '<image_6>' not in problem
        problem = problem.replace('<image_1>', image_placeholder).replace('<image_2>', image_placeholder).replace('<image_3>', image_placeholder).replace('<image_4>', image_placeholder).replace('<image_5>', image_placeholder)
        if not problem.count(image_placeholder) == len(images):
            continue
        new_row = {
            'images': ensure_list(images),
            'data_source': "hiyouga/geometry3k",
            'prompt': ensure_list({'content': problem + format_prompt, 'role': 'user'}),
            'ability': "puzzle",
            'reward_model': {'ground_truth': row['answer'], 'style': 'rule'},
            'extra_info': {'thinking': row.get('pattern', ''), 'problem': problem, 'solution': row['answer']}
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

    # # 如果需要保存修改后的DataFrame
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