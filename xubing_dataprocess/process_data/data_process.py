import os
import sys
import argparse
# from transformers import AutoTokenizer
# from datakit.utils.files import read_mmq_index, read_mmq_recordio
from datakit.utils.files import dump_list_to_jsonl_file, find_all_files, read_jsonl_file

import json
import random
import re

sft_data_paths = []
sft_data_name = {
    'CHART': [
        'ChartQA'
    ],
    'DIAGRAM': [
        'infovqa',
        'mapqa',
    ],
    'GROUNDING': [
        'sharegpt4v_ref'
    ],
    'MATH': [
        'clevr_math_5w',
        'geo3k',
        'geoqa+',
        'iconqa_choose_txt',
        'iconqa_fill_blank',
        'super_clever'
    ],
    'SCIENCE': [
        'mm_ai2d',
        'scienceqa',
        'tqa'
    ]
}

import base64
import io
from typing import List

import numpy as np
from PIL import Image, ImageDraw

test = read_jsonl_file("/mnt/cephfs/bensenliu/wfs/vlmdatasets/pt/laion5b-en/data/0/003965.jsonl")
breakpoint()
# fs = read_jsonl_file("/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_few_shot/geoqa+/data/grammar_correct/02.jsonl")
# cot = read_jsonl_file("/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_cot/geoqa+/data/grammar_correct_vlm_infer_Qwen72B_complexity_vllm_pe/02.jsonl")
cot = read_jsonl_file("/mnt/cephfs/xubingye/wfs/weights/vlm/mmq-pointsv15-20250513e1xubing-20250519e2xubing-soup-sft-hf/vocab.json")
breakpoint()

def decode_base64_image_to_np(base64_image: str) -> np.ndarray:
    """Decode a base64 string of an image to a numpy array.

    Args:
        base64_image (str): The base64 string of the image.

    Returns:
        np.ndarray: The numpy array of the image.
    """
    img_data = base64.b64decode(base64_image)
    img = Image.open(io.BytesIO(img_data))
    return np.array(img)


print("cot")
for item in cot:
    base64_image_dict = item['base64_image']
    for image_name, base64_image in base64_image_dict.items():
        np_image = decode_base64_image_to_np(base64_image)
        height, width = np_image.shape[:2]
        assert height > 28 and width > 28, f"image {image_name} height {height} width {width} is not valid"
print("fs")
for item in fs:
    base64_image_dict = item['base64_image']
    for image_name, base64_image in base64_image_dict.items():
        np_image = decode_base64_image_to_np(base64_image)
        height, width = np_image.shape[:2]
        assert height > 28 and width > 28, f"image {image_name} height {height} width {width} is not valid"

breakpoint()


tt = read_jsonl_file("/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_few_shot/ChartQA/data/grammar_correct/00.jsonl")
qq = read_jsonl_file("/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_few_shot/ChartQA/jsonl/grammar_correct/qwen_2_5_vl_72b_vllm_cot/grammar_correct/00.jsonl")
breakpoint()
all_data_root = "/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results"
aa = read_jsonl_file("/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_few_shot/ChartQA/data/grammar_correct/00.jsonl")
bb = read_jsonl_file("/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_few_shot/ChartQA/jsonl/grammar_correct/qwen_2_5_vl_72b_vllm_cot/grammar_correct/00.jsonl")
bb = read_jsonl_file("/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results/ChartQA.jsonl")

breakpoint()

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


with open("/mnt/cephfs/xubingye/vlm/MMDataKit/workspace/complex_data_to_filter_cot_3B.txt", "r") as f:
    datasets = f.readlines()
datasets = [dataset.strip() for dataset in datasets]

all_data_root = "/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results"

for dataset in datasets:
    print("######## dataset", dataset)
    all_dataset = os.path.join(all_data_root, dataset.split('/')[-1] + ".jsonl")
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
            pattern = r'Question: (.*?)\nStandard answer: '
            match = re.search(pattern, item['conversations'][0]['text'], re.DOTALL)
            assert match
            item['conversations'][0]['text'] = match.group(1)

            # 插入“think step by step”
            for single_word_prompt in single_word_prompts:
                if single_word_prompt in item['conversations'][0]['text']:
                    item['conversations'][0]['text'] = item['conversations'][0]['text'].replace(single_word_prompt, "")
            random_think_prompt = random.choice(think_prompts)
            item['conversations'][0]['text'] += " " + random_think_prompt

            # 这里暂时不进行easy/hard之间的shuffle操作

        dump_list_to_jsonl_file(os.path.join(save_root, jsonl.split("/")[-1]), filtered_data)
    dump_list_to_jsonl_file(os.path.join(save_root, "easy_data.jsonl"), easy_data)

breakpoint()


# root = "/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results"
# save_root = "/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_cot_input"
# jsonls = find_all_files(root, '.jsonl')
# for _idx, jsonl in enumerate(jsonls):
#     data = read_jsonl_file(jsonl)
#     data1 = [{'id': item['id'], 'base64_image': item['base64_image'], 'conversations': item['conversations']} for item in data if item['complexity'] == 'hard']
#     dump_list_to_jsonl_file(os.path.join(save_root, jsonl.split("/")[-1]), data1)

# data2 = read_jsonl_file("/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/CHART/ChartQA/data/grammar_correct_single_word/ChartQA.jsonl")
# breakpoint()  
# dump_list_to_jsonl_file('/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B/infovqa_single_word_sample.jsonl', data[:1000])
# data = read_jsonl_file(jsonls[1])
# dump_list_to_jsonl_file('/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B/sharegpt4v_ref_sample.jsonl', data[:2000])
# data = read_jsonl_file(jsonls[2])
# dump_list_to_jsonl_file('/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B/super_clever_sample.jsonl', data[:1000])
# data = read_jsonl_file(jsonls[3])
# dump_list_to_jsonl_file('/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B/tqa_sample.jsonl', data[:2000])


# for key, value in sft_data_name.items():
#     for cur_dataset_name in value:
#         sft_data_paths.append("/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/" + key + "/" + cur_dataset_name + "/data/grammar_correct")
# Save each path in sft_data_paths to a txt file, one per line
output_txt_path = "/mnt/cephfs/xubingye/vlm/MMDataKit/workspace/complex_data_to_filter_single_word.txt"
# with open(output_txt_path, "w") as f:
#     for path in sft_data_paths:
#         f.write(path + "\n")

# add single word to the end of each conversation
# with open(output_txt_path, "r") as f:
#     datasets = f.readlines()
# datasets = [dataset.strip() for dataset in datasets]
# for dataset in datasets:
#     folder_path = dataset + "_single_word"
#     os.makedirs(folder_path, exist_ok=True)
#     all_jsonls = find_all_files(dataset, "jsonl")
#     data = read_jsonl_file(all_jsonls[0])
#     for item in data:
#         assert len(item['conversations']) == 2, f"item {item['id']} has more than one conversation"
#         item['conversations'][0]['text'] += " Answer the question using a single word or phrase."
#     dump_list_to_jsonl_file(folder_path + "/" + all_jsonls[0].split("/")[-1], data)

#     output_folder = "/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B"
#     os.makedirs(output_folder, exist_ok=True)

#     difficult_all_data = []
#     easy_all_data = []
#     all_data = []
    
#     for jsonl in all_jsonls:
#         data = read_jsonl_file(jsonl)
#         all_data.extend(data)
#     all_data_id_list = list(set([item['id'] for item in all_data]))
#     all_data_id_hash_list = list(set([hash(item['id'] + item['conversations'][0]['text']) for item in all_data]))
#     print("all_jsonls", all_jsonls)
#     print("all_data_id_list", len(all_data_id_list))
#     print("all_data_id_hash_list", len(all_data_id_hash_list))
#     print("all_data", len(all_data), "\n")

#     # assert len(all_data_id_list) == len(all_data), "all_data_id_list length is not equal to all_data length"
#     # assert len(all_data_id_hash_list) == len(all_data), "all_data_id_hash_list length is not equal to all_data length"

#     for diff_jsonl, easy_jsonl in zip(difficult_all_jsonls, easy_all_jsonls):
#         diff_data = read_jsonl_file(diff_jsonl)
#         difficult_all_data.extend(diff_data)
#         easy_data = read_jsonl_file(easy_jsonl)
#         easy_all_data.extend(easy_data)
#         print(len(diff_data), len(easy_data))
    
#     # difficult_all_data_id_list = list(set([item['id'] for item in difficult_all_data]))
#     # difficult_all_data_id_hash_list = list(set([hash(item['id'] + item['conversations'][0]['text']) for item in difficult_all_data]))
#     # print("difficult_all_jsonls", difficult_all_jsonls)
#     # print("difficult_all_data_id_list", len(difficult_all_data_id_list))
#     # print("difficult_all_data_id_hash_list", len(difficult_all_data_id_hash_list))
#     # print("difficult_all_data", len(difficult_all_data), "\n")
#     # easy_all_data_id_list = list(set([item['id'] for item in easy_all_data]))
#     # easy_all_data_id_hash_list = list(set([hash(item['id'] + item['conversations'][0]['text']) for item in easy_all_data]))
#     # print("easy_all_jsonls", easy_all_jsonls)
#     # print("easy_all_data_id_list", len(easy_all_data_id_list))
#     # print("easy_all_data_id_hash_list", len(easy_all_data_id_hash_list))
#     # print("easy_all_data", len(easy_all_data), "\n")

#     difficult_all_data_dict = {}
#     for item in difficult_all_data:
#         difficult_all_data_dict[hash(item['id'] + item['conversations'][0]['text'])] = item
#     easy_all_data_dict = {}
#     for item in easy_all_data:
#         easy_all_data_dict[hash(item['id'] + item['conversations'][0]['text'])] = item

#     for data in all_data:
#         hash_key = hash(data['id'] + data['conversations'][0]['text'])
#         assert hash_key in difficult_all_data_dict, f"hash_key {hash_key} not in difficult_all_data_dict"
#         assert hash_key in easy_all_data_dict, f"hash_key {hash_key} not in easy_all_data_dict"
#         data['difficult_conversations'] = difficult_all_data_dict[hash_key]['conversations']
#         data['easy_conversations'] = easy_all_data_dict[hash_key]['conversations']

#     output_jsonl_path = os.path.join(output_folder, f"{dataset.split('/')[8]}.jsonl")
#     dump_list_to_jsonl_file(output_jsonl_path, all_data)

#     print(dataset, len(all_data), len(difficult_all_data), len(easy_all_data))
