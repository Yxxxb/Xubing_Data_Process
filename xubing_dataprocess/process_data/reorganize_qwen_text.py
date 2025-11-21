import os
import sys
import argparse
from datakit.utils.files import dump_list_to_jsonl_file, find_all_files, read_jsonl_file
from tqdm import tqdm

"""
由于qwen刷出来的数据中conversation是“qwen2.5vl-72b：”的格式，然而打包recordio前需要转换为text用鱼训练，此py用于转换并删除qwen字段
"""

with open("/mnt/cephfs/xubingye/vlm/MMDataKit/workspace/data_to_filter.txt", 'r') as f:
    datasets = f.readlines()
    datasets = [dataset.strip() for dataset in datasets]
    idx = 0
    for dataset in tqdm(datasets):
        print("#############", idx, dataset)
        cur_raw_root_list = dataset + "_vlm_infer_Qwen72B_vllm_12M_reso"
        all_jsonls = find_all_files(cur_raw_root_list, "jsonl")

        save_root = cur_raw_root_list + "_reorganize_text"
        os.makedirs(save_root, exist_ok=True)

        for jsonl in all_jsonls:
            data = read_jsonl_file(jsonl)
            for item in data:
                assert 'qwen2.5vl-72b' in item['conversations'][1]
                item['conversations'][1]['text'] = item['conversations'][1]['qwen2.5vl-72b']
                del item['conversations'][1]['qwen2.5vl-72b']

            dump_list_to_jsonl_file(os.path.join(save_root, jsonl.split("/")[-1]), data)
