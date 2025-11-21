import os
import sys
import argparse
# from transformers import AutoTokenizer
# from datakit.utils.files import read_mmq_index, read_mmq_recordio
from datakit.utils.files import dump_list_to_jsonl_file, find_all_files, read_jsonl_file

import json
import random
import re


single_word_prompts = [
    "Answer the question using a single word or phrase.",
    "Answer the above question or filling in the blank using a single word or phrase.",
    "Answer with the option's letter from the given choices directly.",
    "Answer with the option's letter from the above options directly.",
    "Answer with the letter.",
    "Answer the question using a single word or phrase.",
]

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


# with open("/mnt/cephfs/xubingye/vlm/MMDataKit/workspace/complex_data_to_filter_cot_3B.txt", "r") as f:
with open("/mnt/cephfs/xubingye/vlm/MMDataKit/workspace/complex_data_to_filter_cot_3B_wo_gt.txt", "r") as f:
    datasets = f.readlines()
datasets = [dataset.strip() for dataset in datasets]

# 包含easy和hard标签、以及原始conversations、easy_conv、hard_conv的jsonl文件
all_data_root = "/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results"

for dataset in datasets:
    print("######## dataset", dataset)
    all_dataset = os.path.join(all_data_root, dataset.split('/')[-3] + ".jsonl")
    easy_data = [{'id': item['id'], 'base64_image': item['base64_image'], 'conversations': item['conversations']} for item in read_jsonl_file(all_dataset) if item['complexity'] == 'easy']
    
    # qwen72B打cot标签的路径
    dataset = dataset + "_vlm_infer_Qwen72B_complexity_vllm"
    all_jsonls = find_all_files(dataset, "jsonl")

    save_root = dataset + "_pe"
    os.makedirs(save_root, exist_ok=True)

    for jsonl in all_jsonls:
        data = read_jsonl_file(jsonl)

        # drop none
        filtered_data = []
        for item in data:
            assert "conversations" in item and len(item["conversations"]) == 2
            if item["conversations"][1]["qwen2.5vl-72b"] is not None:
                filtered_data.append(item)

        for item in filtered_data:
            # reorganize "qwen" -> "text"
            assert 'qwen2.5vl-72b' in item['conversations'][1]
            item['conversations'][1]['text'] = item['conversations'][1]['qwen2.5vl-72b']
            del item['conversations'][1]['qwen2.5vl-72b']

            # 通过Qwen72BCoT的Question获取原本的Question
            # pattern = r'Question: (.*?)\nStandard answer: '
            pattern = r'Question: (.*)'
            match = re.search(pattern, item['conversations'][0]['text'], re.DOTALL)
            assert match
            item['conversations'][0]['text'] = match.group(1)

            # 插入“think step by step”
            for single_word_prompt in single_word_prompts:
                if single_word_prompt in item['conversations'][0]['text']:
                    item['conversations'][0]['text'] = item['conversations'][0]['text'].replace(single_word_prompt, "")
            random_think_prompt = random.choice(think_prompts)
            item['conversations'][0]['text'] += random_think_prompt

            # 这里暂时不进行easy/hard之间的shuffle操作

        dump_list_to_jsonl_file(os.path.join(save_root, jsonl.split("/")[-1]), filtered_data)
    dump_list_to_jsonl_file(os.path.join(save_root, "easy_data.jsonl"), easy_data)
