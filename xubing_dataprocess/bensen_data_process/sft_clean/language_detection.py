import re
import os
from tqdm import tqdm
from datakit.utils.files import (read_jsonl_file,
                                 dump_list_to_jsonl_file)
from datakit.utils.distributed import dist_split_files
from datakit.utils.mp import multi_process_with_append
import time


points_sft = [
    'mm_ai2d',
    'docvqa',
    'dvqa',
    'geoqa+',
    'allava_cap',
    'iconqa_choose_txt',
    'iconqa_fill_blank',
    'infovqa',
    'kvqa',
    'gpt4v',
    'llavar',
    'scienceqa',
    'sharegpt4v',
    'stvqa',
    'super_clever',
    'textvqa',
    'tqa',
    'vsr',
    'icdar_2015',
    'lima',
    'alpaca-gpt4',
    'openhermes2.5',
    'mini-gemini',
    'hme100k',
    'tabwp_cot',
    'geo3k',
    'clevr_math_5w',
    'poie',
    'lvis_instruct4v_cap',
    'MetaMathQA',
    'MathInstruct',
    'orca-math-word-problems-200k',
    'math',
    '500k-atlas-math',
    'gpt4o-complex-20240809-en',
    'MathV360K',
    'mapqa'
]


def contains_mixed_languages(s):
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    english_pattern = re.compile(r'[a-zA-Z]')

    contains_chinese = bool(chinese_pattern.search(s))
    contains_english = bool(english_pattern.search(s))

    return contains_chinese and contains_english


def construct_conversations(conversations):
    conversation_str = ''
    for conversation in conversations:
        role = conversation['role']
        text = conversation['text']
        conversation_str += f'{role}: {text}\n'
    return conversation_str.strip()


def detect_language(item):
    conversations = item['conversations']
    conversation_str = construct_conversations(conversations)
    if contains_mixed_languages(conversation_str):
        return [None, item]
    else:
        return [item, None]


if __name__ == '__main__':
    root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2'
    points_sft_cur_rank = dist_split_files(points_sft)
    failed_datasets = []
    success_datasets = []
    mixed_lan_datasets = []
    for dataset in tqdm(points_sft_cur_rank):
        print(f'processing {dataset}...')
        cur_file = os.path.join(root, dataset, 'data', 'grammar_correct',
                                f'{dataset}.jsonl')
        single_lan_output_file = os.path.join(root, dataset, 'data', 
                                              'grammar_correct_single_lan', 
                                              f'{dataset}.jsonl')
        mixed_lan_output_file = os.path.join(root, dataset, 'data', 
                                             'grammar_correct_mixed_lan', 
                                             f'{dataset}.jsonl')
        # if os.path.exists(single_lan_output_file) and os.path.exists(mixed_lan_output_file):
        #     print('file exists, skipped')
        #     continue
        if not os.path.exists(cur_file):
            print(f'file not found for {dataset}, skipped')
            failed_datasets.append(dataset)
            continue
        os.makedirs(os.path.dirname(single_lan_output_file), exist_ok=True)
        os.makedirs(os.path.dirname(mixed_lan_output_file), exist_ok=True)
        try:
            data = read_jsonl_file(cur_file)
            results = multi_process_with_append(detect_language, data, 64)
            single_lan_results = [item[0] for item in results if item[0] is not None]
            mixed_lan_results = [item[1] for item in results if item[1] is not None]
            dump_list_to_jsonl_file(single_lan_output_file, single_lan_results)
            dump_list_to_jsonl_file(mixed_lan_output_file, mixed_lan_results)
            success_datasets.append(dataset)
            if len(mixed_lan_results) > 0:
                mixed_lan_datasets.append(dataset)
        except Exception as e:
            print(f'failed to process {dataset}, error: {e}')
            failed_datasets.append(dataset)
    print(f'mixed language datasets: {mixed_lan_datasets}')
    print(f'success datasets: {success_datasets}')
    print(f'failed datasets: {failed_datasets}')
    time.sleep(1000000)