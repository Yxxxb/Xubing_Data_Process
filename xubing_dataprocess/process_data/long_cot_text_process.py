import os
import sys
import argparse
# from transformers import AutoTokenizer
# from datakit.utils.files import read_mmq_index, read_mmq_recordio
from datakit.utils.files import dump_list_to_jsonl_file, find_all_files, read_jsonl_file, read_parquet_file

import json
import random
import re

from tqdm import tqdm

# demo_jsonl = "/mnt/cephfs/xubingye/wfs/datasets/rl-category/LONG-COT-TEXT/AM-DeepSeek-R1-Distilled-1.4M/jsonl/grammar_correct/openmathreasoning_text_long_cot/grammar_correct/raw_0.jsonl"
# demo_jsonl_data1 = read_jsonl_file(demo_jsonl)
# demo_jsonl = "/mnt/cephfs/xubingye/wfs/datasets/rl-category/LONG-COT-TEXT/OpenMathReasoning/jsonl/grammar_correct/openmathreasoning_text_long_cot/grammar_correct/raw_0.jsonl"
# demo_jsonl_data2 = read_jsonl_file(demo_jsonl)
# breakpoint()

# c:  8028352
# c_over_16k:  237599
# c_over_16k / c:  0.02959499035418477
# avg_len:  6321.537517039612

think_prompts = [
    "Think step by step.",
    "Reason through the problem one step at a time.",
    "Break down your answer into logical steps.",
    "Explain your thought process in detail, step by step.",
    "Work through the solution gradually, showing each stage.",
    "Solve the problem by outlining each step clearly.",
    "Approach the question methodically, detailing every step.",
    "Provide your answer by reasoning sequentially.",
    "Walk through your reasoning process stepwise.",
    "Lay out your solution in a clear, step-by-step manner.",
    "Describe each stage of your reasoning as you solve the problem."
]

"""
# OpenMathReasoning
# <think></think>
cot_path = "/mnt/cephfs/xubingye/wfs/datasets/rl-category/OpenMathReasoning/data"
parquet_files = find_all_files(cot_path, extension=".parquet")

idx = 0
# inference_mode not None
rl_raw_save_list = []
# inference_mode None
sft_raw_save_list = []
for file in tqdm(parquet_files):
    file_data = read_parquet_file(file)
    for item in file_data:
        if item['generated_solution'] == 'n/a':
            assert item['inference_mode'] == 'n/a' and item['expected_answer'] is not None
            answer = item['expected_answer']
            raw_item = {
                'id': f'OpenMathReasoning{idx}',
                'base64_image': {},
                'conversations': [
                    {'role': 'user', 'text': item['problem']},
                    {'role': 'assistant', 'text': answer}
                ],
            }
            sft_raw_save_list.append(raw_item)
        else:
            assert item['inference_mode'] in ['cot', 'tir', 'genselect']
            answer = item['generated_solution']
            raw_item = {
                'id': f'OpenMathReasoning{idx}',
                'base64_image': {},
                'conversations': [
                    {'role': 'user', 'text': item['problem']},
                    {'role': 'assistant', 'text': answer}
                ],
            }
            rl_raw_save_list.append(raw_item)
        
        idx += 1

print(f"rl_raw_save_list: {len(rl_raw_save_list)}")
print(f"sft_raw_save_list: {len(sft_raw_save_list)}")

save_path = "/mnt/cephfs/xubingye/wfs/datasets/rl-category/LONG-COT-TEXT/OpenMathReasoning/data/grammar_correct"
batch_size = 100000
for i in range(0, len(rl_raw_save_list), batch_size):
    batch = rl_raw_save_list[i:i+batch_size]
    batch_index = i // batch_size
    filename = f'raw_{batch_index}.jsonl'
    dump_list_to_jsonl_file(os.path.join(save_path, filename), batch)

save_path = "/mnt/cephfs/xubingye/wfs/datasets/rl-category/LONG-COT-TEXT/OpenMathReasoning-wo-cot/data/grammar_correct"
batch_size = 100000
for i in range(0, len(sft_raw_save_list), batch_size):
    batch = sft_raw_save_list[i:i+batch_size]
    batch_index = i // batch_size
    filename = f'raw_{batch_index}.jsonl'
    dump_list_to_jsonl_file(os.path.join(save_path, filename), batch)

# print(f"cot: {cot_c}, tir: {tir_c}, gen: {gen_c}, na: {na_c}")
# print(f"total: {cot_c + tir_c + gen_c + na_c}")
# print(f"parquet files: {len(parquet_files)}")
# cot: 3201061, tir: 1718466, gen: 0, na: 758790
# total: 5678317

# c:  5485147
# c_over_16k:  195047
# c_over_16k / c:  0.03555911992878222
# a_len / c:  7405.91702300777

"""

"""
# AM-DeepSeek-R1-Distilled-1.4M
# <think></think> <answer></answer>
cot_path = "/mnt/cephfs/xubingye/wfs/datasets/rl-category/AM-DeepSeek-R1-Distilled-1.4M"
jsonl_files = find_all_files(cot_path, extension=".jsonl")

idx = 0
rl_raw_save_list = []
for file in tqdm(jsonl_files):
    if 'am_0.9M_sample_1k.jsonl' in file:
        continue
    file_data = read_jsonl_file(file)
    for item in file_data:
        assert len(item['messages']) == 2
        raw_item = {
            'id': f'AM-DeepSeek-R1-Distilled-1.4M-{idx}',
            'base64_image': {},
            'conversations': [
                {'role': 'user', 'text': item['messages'][0]['content'].replace('<img>', 'img').replace('<img/>', 'img')},
                {'role': 'assistant', 'text': item['messages'][1]['content'].replace('<answer>', '').replace('</answer>', '').replace('<img>', 'img').replace('<img/>', 'img')}
            ],
        }
        rl_raw_save_list.append(raw_item)
        
        idx += 1

print(f"rl_raw_save_list: {len(rl_raw_save_list)}")

save_path = "/mnt/cephfs/xubingye/wfs/datasets/rl-category/LONG-COT-TEXT/AM-DeepSeek-R1-Distilled-1.4M/data/grammar_correct"
batch_size = 100000
for i in range(0, len(rl_raw_save_list), batch_size):
    batch = rl_raw_save_list[i:i+batch_size]
    batch_index = i // batch_size
    filename = f'raw_{batch_index}.jsonl'
    dump_list_to_jsonl_file(os.path.join(save_path, filename), batch)

# c:  1143205
# c_over_16k:  35785
# c_over_16k / c:  0.03130234734802594
# a_len / c:  5161.584507590502
"""


# OpenThoughts2-1M
# <think></think>
cot_path = "/mnt/cephfs/xubingye/wfs/datasets/rl-category/OpenThoughts2-1M/data"
jsonl_files = find_all_files(cot_path, extension=".parquet")

idx = 0
rl_raw_save_list = []
for file in tqdm(jsonl_files):
    file_data = read_parquet_file(file)
    for item in file_data:
        assert len(item['conversations']) == 2
        raw_item = {
            'id': f'OpenThoughts2-1M-{idx}',
            'base64_image': {},
            'conversations': [
                {'role': 'user', 'text': item['conversations'][0]['value'].replace('<img>', 'img').replace('<img/>', 'img')},
                {'role': 'assistant', 'text': item['conversations'][1]['value'].replace('<img>', 'img').replace('<img/>', 'img')}
            ],
        }
        rl_raw_save_list.append(raw_item)
        
        idx += 1

print(f"rl_raw_save_list: {len(rl_raw_save_list)}")

save_path = "/mnt/cephfs/xubingye/wfs/datasets/rl-category/LONG-COT-TEXT/OpenThoughts2-1M/data/grammar_correct"
# save_path += "_w_think_prompt"
batch_size = 100000
for i in range(0, len(rl_raw_save_list), batch_size):
    batch = rl_raw_save_list[i:i+batch_size]
    batch_index = i // batch_size
    filename = f'raw_{batch_index}.jsonl'
    dump_list_to_jsonl_file(os.path.join(save_path, filename), batch)

# c:  1400000
# c_over_16k:  6767
# c_over_16k / c:  0.004833571428571428
# a_len / c:  3020.1682928571427
