import os
import sys
import argparse
# from transformers import AutoTokenizer
# from datakit.utils.files import read_mmq_index, read_mmq_recordio
from datakit.utils.files import dump_list_to_jsonl_file, find_all_files, read_jsonl_file
import re
from tqdm import tqdm



jsonls = [
    '/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B/ChartQA_single_word.jsonl', 
    '/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B/clevr_math_5w_single_word.jsonl', 
    '/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B/iconqa_fill_blank.jsonl', 
    '/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B/infovqa_single_word.jsonl', 
    # '/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B/infovqa_single_word_sample.jsonl', 
    '/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B/geo3k.jsonl', 
    '/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B/iconqa_choose_txt.jsonl', 
    '/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B/mm_ai2d.jsonl', 
    '/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B/scienceqa.jsonl', 
    '/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B/tqa.jsonl',
    # '/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B/tqa_sample.jsonl',
    '/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B/geoqa+.jsonl', 
    '/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B/super_clever.jsonl', 
    # '/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B/super_clever_sample.jsonl', 
]

qa_datasets = [
    'ChartQA',
    'clevr_math_5w',
    'iconqa_fill_blank',
    'infovqa'
]

choice_datasets = [
    'geo3k',
    'iconqa_choose_txt',
    'mm_ai2d',
    'scienceqa',
    'tqa'
]

regular_datasets = [
    'geoqa+',  # 中文
    'super_clever'
]

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def normalize_math_expr(s):
    """
    保留数学表达式的字母、数字、根号、分数线、加减乘除、括号等常见符号
    """
    # 允许的字符：字母、数字、常见数学符号
    return re.sub(r'[^a-zA-Z0-9\+\-\*/\(\)\[\]\{\}√∑∫\.,:：=<>％%．·、\\]', '', s)

def contains_solution_process(s):
    """
    判断字符串是否包含解题过程的特征
    """
    # 过长、包含“解”、“故选”、“：”等
    if len(s) > 50:
        return True
    keywords = ['解', '故选']
    for kw in keywords:
        if kw in s:
            return True
    return False

def extract_choice_from_text(s):
    """
    从文本中提取选择题选项，优先从结尾向前找，
    """
    # 只考虑英文字母a-d，且后面可能跟着点、空格、句号等
    # match = re.search(r'([a-d])[\.\s，,。．]*$', s)
    # 只考虑英文字母a-d，且后面可能跟着任意符号
    match = re.search(r'([a-z])\W*$', s)
    if match:
        return match.group(1)
    # 也可能是“选：C”或“选C”
    match = re.search(r'选[:：]?\s*([a-z])', s)
    if match:
        return match.group(1)
    return None

def match_answer_qa(gt, op):
    """
    输入：   gt - 用户的QA题答案
            op - 正确的QA题答案
    输出：   True - 匹配
            False - 不匹配
    功能：   通过正则表达式匹配QA题答案
    """
    if gt == None or op == None:
        return False
    
    gt = normalize_math_expr(gt.strip().lower())
    op = normalize_math_expr(op.strip().lower())

    if abs(len(gt) - len(op)) > 10:
        return False

    if gt == op:
        return True

    # 去除所有非字母数字再比较
    def alphanum(s):
        return re.sub(r'[^a-z0-9]', '', s)
    if alphanum(gt) == alphanum(op):
        return True

    # 如果都是纯数字 那么float匹配
    # 如果任意一个包含字符/后缀/逗号/数学符号，去除非数学符号后字符串匹配
    gt_num = re.match(r'^([0-9]+(?:\.[0-9]+)?)', gt)
    op_num = re.match(r'^([0-9]+(?:\.[0-9]+)?)', op)
    if is_number(gt) and is_number(op):
        if float(gt) == float(op):
            return True
        else:
            return False
    elif gt_num and op_num:
        # gt_num_val = re.sub(r'[^0-9\+\-\*/\(\)\[\]\{\}√∑∫\.,:：=<>％%．·、\\]', '', gt)
        # op_num_val = re.sub(r'[^0-9\+\-\*/\(\)\[\]\{\}√∑∫\.,:：=<>％%．·、\\]', '', op)
        gt_num_val = re.sub(r'[^0-9\+\-\*/\(\)\[\]\{\}√∑∫\.,]', '', gt)
        op_num_val = re.sub(r'[^0-9\+\-\*/\(\)\[\]\{\}√∑∫\.,]', '', op)
        if is_number(gt_num_val) and is_number(op_num_val) and float(gt_num_val) == float(op_num_val):
            return True
        elif gt_num_val == op_num_val:
            return True
        else:
            return False

    # 非数字，允许后缀g、%、.、标点等
    # 允许gt和op互为前缀或后缀（如serif fonts <-> serif）
    if gt in op or op in gt:
        return True

    return False

def match_answer_choice(gt, op):
    """
    输入：   gt - 用户的选择题答案
            op - 正确的选择题答案
    输出：   True - 匹配
            False - 不匹配
    功能：   通过正则表达式匹配选择题答案
            最基本匹配：gt和op相等
            只删字符串后的标点符号和空格
            剩余严格匹配
    """
    if gt == None or op == None:
        return False
    
    gt = normalize_math_expr(gt.strip().lower())
    op = normalize_math_expr(op.strip().lower())

    if gt == op:
        return True

    # 允许gt和op去除常见后缀后再比较
    suffixes = ['.', '。', '，', ',', ' ']
    def strip_suffix(s):
        for suf in suffixes:
            if s.endswith(suf):
                s = s[:-len(suf)]
        return s

    if strip_suffix(gt) == strip_suffix(op):
        return True
    elif len(op) > 5:
        if extract_choice_from_text(op) == gt:
            return True

    return False

def match_answer_geoqa_plus(gt, op):
    if gt == None or op == None:
        return False
    
    gt = gt.strip().lower()
    op = op.strip().lower()

    # gt包含解题过程，直接认定困难
    if contains_solution_process(gt):
        return False

    # gt为选择题选项
    gt_choice = extract_choice_from_text(gt)
    op_choice = extract_choice_from_text(op)
    if gt_choice and op_choice:
        return gt_choice == op_choice

    match = re.search(r'故答案为[:：]\s*(.*)', op)
    if match:
        op = match.group(1)

    # 数字匹配
    return match_answer_qa(gt, op)

def match_answer_super_clever(gt, op):
    gt = normalize_math_expr(gt.strip().lower()).replace("true", "yes").replace("false", "no")
    op = normalize_math_expr(op.strip().lower()).replace("true", "yes").replace("false", "no")
    return match_answer_qa(gt, op)

if __name__ == "__main__":
    easy_count = 0
    hard_count = 0
    for _idx, jsonl in enumerate(jsonls):
        dataset_name = jsonl.split("/")[-1].split(".jsonl")[0].split("_sample")[0].split("_single_word")[0]
        # if _idx != 9: continue
        data = read_jsonl_file(jsonl)

        cur_easy_count = 0
        cur_hard_count = 0
        contains_solution_process_count = 0

        if dataset_name in qa_datasets:
            for idx, item in enumerate(data):
                assert len(item['conversations']) == 2
                easy_match = match_answer_qa(item['conversations'][1]['text'], item['easy_conversations'][1]['qwen2.5vl-3b'])
                # easy_match = match_answer_qa(item['conversations'][1]['text'], item['difficult_conversations'][1]['qwen2.5vl-72b'])
                if easy_match:
                    cur_easy_count += 1
                    item['complexity'] = "easy"
                else:
                    cur_hard_count += 1
                    item['complexity'] = "hard"
                
        elif dataset_name in choice_datasets:
            for idx, item in enumerate(data):
                assert len(item['conversations']) == 2
                easy_match = match_answer_choice(item['conversations'][1]['text'], item['easy_conversations'][1]['qwen2.5vl-3b'])
                # easy_match = match_answer_choice(item['conversations'][1]['text'], item['difficult_conversations'][1]['qwen2.5vl-72b'])
                if easy_match:
                    cur_easy_count += 1
                    item['complexity'] = "easy"
                else:
                    cur_hard_count += 1
                    item['complexity'] = "hard"

        elif dataset_name in regular_datasets:
            if dataset_name == 'super_clever':
                for idx, item in enumerate(data):
                    assert len(item['conversations']) == 2
                    easy_match = match_answer_super_clever(item['conversations'][1]['text'], item['easy_conversations'][1]['qwen2.5vl-3b'])
                    # easy_match = match_answer_super_clever(item['conversations'][1]['text'], item['difficult_conversations'][1]['qwen2.5vl-72b'])
                    if easy_match:
                        cur_easy_count += 1
                        item['complexity'] = "easy"
                        # print("#### match")
                        # print("text: ", item['conversations'][1]['text'], "\nqwen2.5vl-72b: ", item['difficult_conversations'][1]['qwen2.5vl-72b'], "\n")
                    else:
                        cur_hard_count += 1
                        item['complexity'] = "hard"
                        # print("**** not match")
                        # print("text: ", item['conversations'][1]['text'], "\nqwen2.5vl-72b: ", item['difficult_conversations'][1]['qwen2.5vl-72b'], "\n")
                    
            elif dataset_name == 'geoqa+':
                for idx, item in enumerate(data):
                    assert len(item['conversations']) == 2
                    easy_match = match_answer_geoqa_plus(item['conversations'][1]['text'], item['easy_conversations'][1]['qwen2.5vl-3b'])
                    # easy_match = match_answer_geoqa_plus(item['conversations'][1]['text'], item['difficult_conversations'][1]['qwen2.5vl-72b'])
                    if easy_match:
                        cur_easy_count += 1
                        item['complexity'] = "easy"
                    else:
                        cur_hard_count += 1
                        item['complexity'] = "hard"
            else:
                assert False, f"dataset {dataset_name} not in regular_datasets"

        else:
            assert False, f"dataset {dataset_name} not in qa_datasets, choice_datasets or regular_datasets"
        
        easy_count += cur_easy_count
        hard_count += cur_hard_count
        print("########## dataset_name", dataset_name, "easy_count", cur_easy_count, "hard_count", cur_hard_count)

        save_path = "/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results/" + dataset_name + ".jsonl"
        dump_list_to_jsonl_file(save_path, data)

    
    print("########## total_easy_count", easy_count, "total_hard_count", hard_count, "total_count", easy_count + hard_count)
    print("########## total_easy_rate", easy_count / (easy_count + hard_count), "total_hard_rate", hard_count / (easy_count + hard_count))
    print("########## contains_solution_process_count", contains_solution_process_count)


"""
qwen2.5vl-3b
########## dataset_name ChartQA easy_count 18934 hard_count 13785
########## dataset_name clevr_math_5w easy_count 14667 hard_count 33557
########## dataset_name iconqa_fill_blank easy_count 4865 hard_count 6048
########## dataset_name infovqa easy_count 10440 hard_count 13988
########## dataset_name geo3k easy_count 812 hard_count 1239
########## dataset_name iconqa_choose_txt easy_count 10490 hard_count 8379
########## dataset_name mm_ai2d easy_count 364 hard_count 91
########## dataset_name scienceqa easy_count 4491 hard_count 1658
########## dataset_name tqa easy_count 5708 hard_count 3270
########## dataset_name geoqa+ easy_count 31924 hard_count 36221
########## dataset_name super_clever easy_count 130654 hard_count 124184

########## total_easy_count 233349 total_hard_count 242420 total_count 475769
########## total_easy_rate 0.4904670123526333 total_hard_rate 0.5095329876473667


qwen2.5vl-72b
########## dataset_name ChartQA easy_count 24456 hard_count 8263
########## dataset_name clevr_math_5w easy_count 39214 hard_count 9010
########## dataset_name iconqa_fill_blank easy_count 7777 hard_count 3136
########## dataset_name infovqa easy_count 20369 hard_count 4059
########## dataset_name geo3k easy_count 1053 hard_count 998
########## dataset_name iconqa_choose_txt easy_count 15090 hard_count 3779
########## dataset_name mm_ai2d easy_count 427 hard_count 28
########## dataset_name scienceqa easy_count 5496 hard_count 653
########## dataset_name tqa easy_count 7239 hard_count 1739
########## dataset_name geoqa+ easy_count 56198 hard_count 11947
########## dataset_name super_clever easy_count 177688 hard_count 77150

########## total_easy_count 355007 total_hard_count 120762 total_count 475769
########## total_easy_rate 0.7461751396160742 total_hard_rate 0.2538248603839258
########## contains_solution_process_count 11112

"""
