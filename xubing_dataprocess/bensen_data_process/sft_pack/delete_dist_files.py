from datakit.utils.files import find_all_files
import os
from tqdm import tqdm


points_datasets = [
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
    'MathV360K'
]

if __name__ == '__main__':
    root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2'
    for dataset in tqdm(points_datasets):
        dataset_folder = f'{root}/{dataset}/jsonl/qwen2vl-grammar-correct'
        files = find_all_files(dataset_folder, 'jsonl')
        if len(files) == 0:
            continue
        files_to_remove = [f for f in files if 'data_' in f]
        for f in files_to_remove:
            print(f'Removing {f}')
            os.remove(f)
