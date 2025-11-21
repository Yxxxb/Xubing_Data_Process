import os
import sys
import argparse
# from transformers import AutoTokenizer
# from datakit.utils.files import read_mmq_index, read_mmq_recordio
from datakit.utils.files import dump_list_to_jsonl_file, find_all_files, read_jsonl_file

import json
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

# for key, value in sft_data_name.items():
#     for cur_dataset_name in value:
#         sft_data_paths.append("/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/" + key + "/" + cur_dataset_name + "/data/grammar_correct")
# Save each path in sft_data_paths to a txt file, one per line
output_txt_path = "/mnt/cephfs/xubingye/vlm/MMDataKit/workspace/complex_data_to_filter_single_word.txt"
# with open(output_txt_path, "w") as f:
#     for path in sft_data_paths:
#         f.write(path + "\n")


with open(output_txt_path, "r") as f:
    datasets = f.readlines()
datasets = [dataset.strip() for dataset in datasets]
for dataset in datasets:
    difficult_dataset = dataset + "_vlm_infer_Qwen72B_complexity_vllm_single_word"
    easy_dataset = dataset + "_vlm_infer_Qwen3B_complexity_vllm_single_word"
    difficult_all_jsonls = find_all_files(difficult_dataset, "jsonl")
    easy_all_jsonls = find_all_files(easy_dataset, "jsonl")
    all_jsonls = find_all_files(dataset, "jsonl")

    output_folder = "/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B"
    os.makedirs(output_folder, exist_ok=True)

    difficult_all_data = []
    easy_all_data = []
    all_data = []
    
    for jsonl in all_jsonls:
        data = read_jsonl_file(jsonl)
        all_data.extend(data)
    all_data_id_list = list(set([item['id'] for item in all_data]))
    all_data_id_hash_list = list(set([hash(item['id'] + item['conversations'][0]['text']) for item in all_data]))
    print("all_jsonls", all_jsonls)
    print("all_data_id_list", len(all_data_id_list))
    print("all_data_id_hash_list", len(all_data_id_hash_list))
    print("all_data", len(all_data), "\n")

    # assert len(all_data_id_list) == len(all_data), "all_data_id_list length is not equal to all_data length"
    # assert len(all_data_id_hash_list) == len(all_data), "all_data_id_hash_list length is not equal to all_data length"

    for diff_jsonl, easy_jsonl in zip(difficult_all_jsonls, easy_all_jsonls):
        diff_data = read_jsonl_file(diff_jsonl)
        difficult_all_data.extend(diff_data)
        easy_data = read_jsonl_file(easy_jsonl)
        easy_all_data.extend(easy_data)
        print(len(diff_data), len(easy_data))
    
    # difficult_all_data_id_list = list(set([item['id'] for item in difficult_all_data]))
    # difficult_all_data_id_hash_list = list(set([hash(item['id'] + item['conversations'][0]['text']) for item in difficult_all_data]))
    # print("difficult_all_jsonls", difficult_all_jsonls)
    # print("difficult_all_data_id_list", len(difficult_all_data_id_list))
    # print("difficult_all_data_id_hash_list", len(difficult_all_data_id_hash_list))
    # print("difficult_all_data", len(difficult_all_data), "\n")
    # easy_all_data_id_list = list(set([item['id'] for item in easy_all_data]))
    # easy_all_data_id_hash_list = list(set([hash(item['id'] + item['conversations'][0]['text']) for item in easy_all_data]))
    # print("easy_all_jsonls", easy_all_jsonls)
    # print("easy_all_data_id_list", len(easy_all_data_id_list))
    # print("easy_all_data_id_hash_list", len(easy_all_data_id_hash_list))
    # print("easy_all_data", len(easy_all_data), "\n")

    difficult_all_data_dict = {}
    for item in difficult_all_data:
        difficult_all_data_dict[hash(item['id'] + item['conversations'][0]['text'])] = item
    easy_all_data_dict = {}
    for item in easy_all_data:
        easy_all_data_dict[hash(item['id'] + item['conversations'][0]['text'])] = item

    for data in all_data:
        hash_key = hash(data['id'] + data['conversations'][0]['text'])
        assert hash_key in difficult_all_data_dict, f"hash_key {hash_key} not in difficult_all_data_dict"
        assert hash_key in easy_all_data_dict, f"hash_key {hash_key} not in easy_all_data_dict"
        data['difficult_conversations'] = difficult_all_data_dict[hash_key]['conversations']
        data['easy_conversations'] = easy_all_data_dict[hash_key]['conversations']

    output_jsonl_path = os.path.join(output_folder, f"{dataset.split('/')[8]}_single_word.jsonl")
    # breakpoint()
    dump_list_to_jsonl_file(output_jsonl_path, all_data)

    print(dataset, len(all_data), len(difficult_all_data), len(easy_all_data))
