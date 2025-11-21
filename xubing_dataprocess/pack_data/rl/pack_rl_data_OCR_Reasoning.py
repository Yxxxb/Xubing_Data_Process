import pandas as pd
import os
import glob
import numpy as np
import re
from tqdm import tqdm
import datasets
from datasets import Dataset, concatenate_datasets
from datakit.utils.files import dump_list_to_jsonl_file, find_all_files, read_jsonl_file, mem_efficient_read_jsonl_file, read_parquet_file
import json


def ensure_list(x):
    if isinstance(x, dict):
        return [x]
    if isinstance(x, list):
        return x
    return [x]

def index_to_letter(index):
    return chr(ord('A') + index)

import re

def extract_final_answer(answer_str: str) -> str:
    """
    根据回答文本的语言（中文/英文）提取最终答案
    
    参数:
        answer_str: 包含最终答案的回答字符串
        
    返回:
        提取的最终答案文本，若未找到则返回空字符串
    """
    # 中文场景：匹配 "最终的回答是: " 后面的内容
    chinese_match = re.search(r'最终的回答是:\s*(.+)', answer_str, re.IGNORECASE)
    if chinese_match:
        return chinese_match.group(1).strip()
    
    # 英文场景：匹配 "The final answer is" 后面的内容（支持冒号/句号等标点）
    english_match = re.search(r'The final answer is:\s*(.+)', answer_str, re.IGNORECASE)
    if english_match:
        return english_match.group(1).strip()
    
    # 未找到匹配时返回空
    return ""


# pip install pyarrow
parquets_path = "/mnt/cephfs/xubingye/wfs/datasets/rl-category/OCR-Reasoning"
use_val = False

save_path = parquets_path + "/data_verl_parquet_check"
os.makedirs(save_path, exist_ok=True)
with open('/mnt/cephfs/xubingye/wfs/datasets/rl-category/OCR-Reasoning/annotations.json', 'r', encoding='utf-8') as f:
    # 直接加载整个JSON文件（根结构为列表）
    data = json.load(f)

format_prompt = " You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."
image_placeholder = "<image>"

lst = []
datasets_list = []

sample_num = 0
for i, row in enumerate(data):
    assert len(row['image']) == 1
    img_path = "/mnt/cephfs/xubingye/wfs/datasets/rl-category/OCR-Reasoning/images/" + row['image'][0]
    try:
        with open(img_path, 'rb') as f:
            img_bytes = f.read()
    except:
        try:
            with open(img_path.replace('.jpg', '.png'), 'rb') as f:
                img_bytes = f.read()
        except:
            try:
                with open(img_path.replace('.png', '.jpg'), 'rb') as f:
                    img_bytes = f.read()
            except:
                breakpoint()
    answer = extract_final_answer(row['answer'])
    if answer == '':
        breakpoint()
    new_row = {
        'images': ensure_list({'bytes': img_bytes, 'path': img_path}),
        'data_source': "hiyouga/geometry3k",
        'prompt': ensure_list({'content': image_placeholder + row['question'] + f' The format requirement when answering question is \n{row["format"]}\n. ' + f'Please answer in {row["language"]}. ' + format_prompt, 'role': 'user'}),
        'ability': "puzzle",
        'reward_model': {'ground_truth': answer, 'style': 'rule'},
        'extra_info': {'thinking': row.get('pattern', ''), 'problem': row['question'], 'solution': row['answer']}
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