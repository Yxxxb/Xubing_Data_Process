import os
import sys
import argparse
# from transformers import AutoTokenizer
# from datakit.utils.files import read_mmq_index, read_mmq_recordio
from datakit.utils.files import dump_list_to_jsonl_file, find_all_files, read_jsonl_file, mem_efficient_read_jsonl_file, read_parquet_file

import json
import re

js1 = read_jsonl_file("/mnt/cephfs/xubingye/wfs/datasets/rl-category/MMK12/data_verl_parquet_offline_roll/0/train-00000-of-00004.jsonl")
js2 = read_jsonl_file("/mnt/cephfs/xubingye/wfs/datasets/rl-category/MMK12/data_verl_parquet_offline_roll_filter_result/0/train-00000-of-00004.jsonl")
js3 = read_jsonl_file("/mnt/cephfs/xubingye/wfs/datasets/rl-category/MMK12/data_verl_parquet_offline_roll_filter_result_exactly_match/0/train-00003-of-00004.jsonl")
js4 = read_jsonl_file("/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/MATH/geoqa+/data/grammar_correct_split_3k_caption_prompt_vlm_infer_Qwen72B_caption_qwen3_cot/6000.jsonl")
# for item in 
breakpoint()

have_qwen3_32b_caption_content = 0
for i in js4:
    if 'qwen3_32b_caption_content' in i:
        have_qwen3_32b_caption_content += 1
print(f"have_qwen3_32b_caption_content: {have_qwen3_32b_caption_content} / {len(js4)}")

breakpoint()

all_jsonls = find_all_files(
    "/mnt/cephfs/xubingye/wfs/datasets/rl-category/MMK12/data_verl_parquet_offline_roll_filter_result_exactly_match", 
    'jsonl')
pass_num = {}
sum_num = 0
for cur_jsonl in all_jsonls:
    cur_data = read_jsonl_file(cur_jsonl)
    for item in cur_data:
        # for roll_item, cur_match_log in zip(item['offline_roll_results'], item['match_log']):
        #     match = re.search(r'\\boxed{(.*)}', roll_item)
        #     if match:
        #         roll_item = match.group(1)
        #     print(f"roll_item: {roll_item}, cur_match_log: {cur_match_log}")
        # print(f"##### Prompt: {item['prompt'][0]['content']} #####")
        # print(f"##### GT: {item['reward_model']['ground_truth']} #####")
        if item['match_num'] in pass_num:
            pass_num[item['match_num']] += 1
        else:
            pass_num[item['match_num']] = 1
        sum_num += 1

# key都是数字，按照key的大小顺序输出
for key in sorted(pass_num.keys()):
    value = pass_num[key]
    print(f"{key}: {value}, percentage: {value / sum_num}")


"""
0: 3331, percentage: 0.19904391992829398
1: 2522, percentage: 0.1507021213026591
2: 1765, percentage: 0.10546758291006872
3: 1373, percentage: 0.08204362115327159
4: 1147, percentage: 0.06853899014042426
5: 906, percentage: 0.05413803406035256
6: 834, percentage: 0.04983567373767553
7: 701, percentage: 0.04188825814161936
8: 666, percentage: 0.03979683298476248
9: 591, percentage: 0.03531520764864057
10: 506, percentage: 0.03023603226770242
11: 453, percentage: 0.02706901703017628
12: 423, percentage: 0.025276366895727518
13: 379, percentage: 0.022647146698536003
14: 464, percentage: 0.027726322079474158
15: 402, percentage: 0.024021511801613386
16: 272, percentage: 0.01625336121900209
"""