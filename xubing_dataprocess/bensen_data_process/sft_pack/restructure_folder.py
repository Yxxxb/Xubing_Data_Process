from datakit.utils.files import find_all_files
import os
from tqdm import tqdm


points_sft = [
    # 'mm_ai2d',
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
    'MathV360K'
]


if __name__ == '__main__':
    root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2'
    for sft in tqdm(points_sft):
        data_folder = os.path.join(root, sft, 'data')
        grammar_sub_folders = os.path.join(data_folder, 'grammar_error')
        correct_sub_folders = os.path.join(data_folder, 'grammar_correct')
        all_jsonls = find_all_files(data_folder, 'jsonl')
        if len(all_jsonls) == 0:
            continue
        correct_jsonls = [f for f in all_jsonls if 'problem' not in f]
        grammar_error_jsonls = [f for f in all_jsonls if 'problem_data' in f]
        concat_file = os.path.join(data_folder, 'problems.jsonl')
        if os.path.exists(concat_file):
            print(f'delete {concat_file}')
            os.remove(concat_file)
        os.makedirs(grammar_sub_folders, exist_ok=True)
        os.makedirs(correct_sub_folders, exist_ok=True)
        for f in correct_jsonls:
            os.system(f'mv {f} {correct_sub_folders}')
        for f in grammar_error_jsonls:
            os.system(f'mv {f} {grammar_sub_folders}')


