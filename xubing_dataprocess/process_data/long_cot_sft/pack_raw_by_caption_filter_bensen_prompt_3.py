import os
import sys
import argparse
# from transformers import AutoTokenizer
# from datakit.utils.files import read_mmq_index, read_mmq_recordio
from datakit.utils.files import dump_list_to_jsonl_file, find_all_files, read_jsonl_file, mem_efficient_read_jsonl_file, read_parquet_file, find_all_files_multi_folders

import json
from tqdm import tqdm
import random


system_prompt = {'role': 'system', 'text': 'You are an intelligent assistant that needs to obtain information through multiple tool calls to solve complex problems. Please follow the process below:\n\n1. When encountering a question that requires external information, proactively call the given tools (such as knowledge retrieval tools, calculators, etc.)\n2. You can make multiple tool calls or none at all\n3. Please try to use existing tools to help the user answer\n4. After each tool call, analyze the returned results and decide the next action:\n    - If information is insufficient → continue calling tools\n    - If information is conflicting → continue calling tools\n    - If information is complete → proceed to the final answer stage\n5. Currently, you can only perform text searches, not image searches, so you cannot perform searches like "what is the object in the picture"\n6. The language you use to answer should be consistent with the language of the user\'s question. For example, if the user\'s question is in English, you should answer in English; if the user\'s question is in Chinese, you should answer in Chinese.\n7. When you have confirmed the user\'s question answer based on tool calls, please answer in the following format: \n    - Exact Answer: <answer>Final answer</answer> \n    - Confidence: <confidence>Confidence percentage based on information completeness and tool reliability</confidence>\n\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{"function": {"description": "Get search result for a query.", "name": "wxg_search", "parameters": {"properties": {"query": {"description": "The query for search", "type": "string"}}, "required": ["query"], "type": "object"}}, "type": "function"}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>\n'}
input_paths = [
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/MATH/clevr_math_5w/data/grammar_correct_split_3k_caption_prompt_vlm_infer_Qwen72B_caption_qwen3_cot_filter_raw",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/MATH/geo3k/data/grammar_correct_split_3k_caption_prompt_vlm_infer_Qwen72B_caption_qwen3_cot_filter_raw",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/MATH/geoqa+/data/grammar_correct_split_3k_caption_prompt_vlm_infer_Qwen72B_caption_qwen3_cot_filter_raw",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/MATH/iconqa_choose_txt/data/grammar_correct_split_3k_caption_prompt_vlm_infer_Qwen72B_caption_qwen3_cot_filter_raw",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/MATH/iconqa_fill_blank/data/grammar_correct_split_3k_caption_prompt_vlm_infer_Qwen72B_caption_qwen3_cot_filter_raw",
    "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/MATH/super_clever/data/grammar_correct_split_3k_caption_prompt_vlm_infer_Qwen72B_caption_qwen3_cot_filter_raw",
]
# bensen = read_jsonl_file('/mnt/cephfs/bensenliu/wfs/vlmdatasets/rl/culture/movie_posters-100k/raw_final/stable.jsonl')
# xb = read_jsonl_file('/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/MATH/clevr_math_5w/data/grammar_correct_split_3k_caption_prompt_vlm_infer_Qwen72B_caption_qwen3_cot_filter_raw/3000.jsonl')
# with open(f"/mnt/cephfs/xubingye/vlm/rl/v0528_rl/checkpoints/wepoints_rl/wepoints_rl_xubing_20250731e3/latest_checkpointed_iteration.txt", 'r', encoding='utf-8') as file:
#     # 读取一行内容（strip()用于去除首尾的空白字符，如换行符、空格等）
#     last_step = file.readline().strip()

for path in tqdm(input_paths):
    save_path = path.replace(
        'grammar_correct_split_3k_caption_prompt_vlm_infer_Qwen72B_caption_qwen3_cot_filter_raw',
        'grammar_correct_split_3k_caption_prompt_vlm_infer_Qwen72B_caption_qwen3_cot_filter_search_prompt_raw')
    os.makedirs(save_path, exist_ok=True)
    files = find_all_files(path, 'jsonl')
    for file in tqdm(files):
        jsonl_save_path = file.replace(path, save_path)
        data = read_jsonl_file(file)
        save_list = []
        for item in data:
            item['conversations'].insert(0, system_prompt)
            item['conversations'][-1]['text'] += f'  \nConfidence: <confidence>{random.randint(85, 100)}%</confidence>'
            save_list.append(item)
        dump_list_to_jsonl_file(jsonl_save_path, save_list)

breakpoint()