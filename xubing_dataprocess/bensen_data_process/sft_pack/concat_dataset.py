from datakit.utils.files import (read_jsonl_file,
                                 dump_list_to_jsonl_file,
                                 find_all_files)
from datakit.utils.mp import multi_process_with_append
import os


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
    'MathV360K'
]

root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2'


def concat_dataset(dataset_name):
    output_file = f'{root}/{dataset_name}/data/grammar_correct/{dataset_name}.jsonl'
    if os.path.exists(output_file):
        return
    data_folder = f'{root}/{dataset_name}/data/grammar_correct'
    if not os.path.exists(data_folder):
        return
    files = find_all_files(data_folder, 'jsonl')
    if len(files) == 0:
        return
    results = []
    for file in files:
        data = read_jsonl_file(file)
        results.extend(data)
    dump_list_to_jsonl_file(output_file, results)
    print(f'Concat {dataset_name} done')


if __name__ == '__main__':
    multi_process_with_append(concat_dataset, points_sft, num_workers=5)
    print('Done')
