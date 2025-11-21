import os
import sys
import argparse
# from transformers import AutoTokenizer
# from datakit.utils.files import read_mmq_index, read_mmq_recordio
from datakit.utils.files import dump_list_to_jsonl_file, find_all_files, read_jsonl_file, mem_efficient_read_jsonl_file, read_parquet_file, find_all_files_multi_folders

import json
from tqdm import tqdm

with open(f"/mnt/cephfs/xubingye/vlm/rl/v0528_rl/checkpoints/wepoints_rl/wepoints_rl_xubing_20250731e3/latest_checkpointed_iteration.txt", 'r', encoding='utf-8') as file:
    # 读取一行内容（strip()用于去除首尾的空白字符，如换行符、空格等）
    last_step = file.readline().strip()

breakpoint()
"""
/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/CHART/ChartQA/data/grammar_correct 32719
/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/DIAGRAM/infovqa/data/grammar_correct 24428
/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/DIAGRAM/mapqa/data/grammar_correct 24720 先不做
/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/GROUNDING/sharegpt4v_ref/data/grammar_correct 79622 
/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/MATH/clevr_math_5w/data/grammar_correct 48224
/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/MATH/geo3k/data/grammar_correct 2051
/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/MATH/geoqa+/data/grammar_correct 68145
/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/MATH/iconqa_choose_txt/data/grammar_correct 18869
/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/MATH/iconqa_fill_blank/data/grammar_correct 10913
/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/MATH/super_clever/data/grammar_correct 254838
/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/SCIENCE/mm_ai2d/data/grammar_correct 455
/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/SCIENCE/scienceqa/data/grammar_correct 6149
/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/SCIENCE/tqa/data/grammar_correct 8978
/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/MATH/MathV360K/data/grammar_correct 325535 先不做
580111
"""

# xb = read_jsonl_file("/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/MATH/geoqa+/data/grammar_correct_split_3k_caption_prompt/18000.jsonl")
# cot = read_jsonl_file("/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/MATH/geoqa+/data/grammar_correct_split_3k_caption_prompt_vlm_infer_Qwen72B_caption_qwen3_cot/18000.jsonl")
# cotf = read_jsonl_file("/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/MATH/iconqa_choose_txt/data/grammar_correct_split_3k_caption_prompt_vlm_infer_Qwen72B_caption_qwen3_cot_filter/9000.jsonl")
# # cot = read_jsonl_file("/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/MATH/geoqa+/data/grammar_correct_split_3k_caption_prompt_vlm_infer_Qwen72B_caption_qwen3_cot/12000.jsonl")
# breakpoint()

match_count = 0
miss_count = 0
invalid_count = 0
format_prompt = " You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."
output_txt_path = "/mnt/cephfs/xubingye/vlm/MMDataKit/workspace/complex_data_to_filter_caption.txt"
with open(output_txt_path, "r") as f:
    datasets = f.readlines()
datasets = [dataset.strip() + '_vlm_infer_Qwen72B_caption_qwen3_cot_filter' for dataset in datasets]
all_jsonls = find_all_files_multi_folders(datasets, 'jsonl')
for jsonl_path in tqdm(all_jsonls):
    jsonl_save_path = jsonl_path.replace(
        'grammar_correct_split_3k_caption_prompt_vlm_infer_Qwen72B_caption_qwen3_cot_filter',
        'grammar_correct_split_3k_caption_prompt_vlm_infer_Qwen72B_caption_qwen3_cot_filter_raw')
    os.makedirs(os.path.dirname(jsonl_save_path), exist_ok=True)
    cur_data = read_jsonl_file(jsonl_path)
    save_list = []
    for item in cur_data:
        if item['is_cot_answer_match_gt'] == 0:
            miss_count += 1
            continue
        elif item['is_cot_answer_match_gt'] == -1:
            invalid_count += 1
            continue
        match_count += 1
        cur_prompt = item["conversations"][0]['text'].replace("You need to generate a **detailed** description of the image based on the image and the question.\n The description needs to be as comprehensive as possible, focusing on the **overall content of the image** and the **details of all the objects**. Your description will be used to answer the question provided, so it is also important that your description contains as much detail as possible about what is involved in answering the question.\n\n For example, for images rich in text and tables, you need to extract the entire content. For images containing multiple objects, you need to give not only a detailed description of each object, but also a description of the image as a whole and the positional relationships between the objects. For mathematical graphs, you need to give a detailed description of the mathematical graph in the context of the problem as much as possible.\n\n The question is '", "")
        cur_prompt = cur_prompt.replace("'.\n\n Please provide description for the image below as requested: ", "")
        cur_prompt += format_prompt
        cur_item = {
            'id': item['id'],
            'base64_image': item['base64_image'],
            'conversations': [
                {'role': 'user', 'text': cur_prompt.replace('<img>', 'img').replace('<img/>', 'img')},
                {'role': 'assistant', 'text': item['qwen3_32b_caption_cot'].replace('<img>', 'img').replace('<img/>', 'img')}
            ],
        }
        save_list.append(cur_item)
    # breakpoint()
    # dump_list_to_jsonl_file(jsonl_save_path, save_list)

print(f"match_count: {match_count}, miss_count: {miss_count}, invalid_count: {invalid_count}")
# breakpoint()
