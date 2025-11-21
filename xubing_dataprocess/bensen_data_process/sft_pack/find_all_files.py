from datakit.utils.files import find_all_files
from data_distribution import data_distribution


SUBSET = 'grammar_correct'
llm = 'qwen-2-5'
include_datasets = [
    # 'mm_ai2d',
    # 'scienceqa',
    # 'tqa',
    # 'sharegpt4v_choice_cn',
    # 'sharegpt4v_qa_cn',
    # 'sharegpt4v_llava158k',
    # 'Mantis-Instruct-spot-the-diff',
    # 'Mantis-Instruct-nlvr2',
    # 'Mantis-Instruct-contrastive_caption',
    # 'Mantis-Instruct-birds-to-words',
    # 'Mantis-Instruct-lrv_multi',
    # 'Mantis-Instruct-iconqa',
    # 'Mantis-Instruct-multi_vqa',
    # 'Mantis-Instruct-dreamsim',
    'coco'
]
points = []
for category, dataset_names in data_distribution.items():
    for dataset_name in dataset_names:
        if dataset_name not in include_datasets:
            continue
        points.append(f'{category}/{dataset_name}')

if __name__ == '__main__':
    print(f'num of points datasets: {len(points)}')
    root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category'
    non_exist_files = []
    results = []
    for filename in points:
        folder = f'{root}/{filename}/recordio/qwen2vl-{SUBSET}-{llm}'
        try:
            data_file = find_all_files(folder, 'data.recordio')[0]
        except Exception as e:
            non_exist_files.append(filename)
            print(f'{filename} not found: {e}')
            continue
        results.append(data_file)
    print(f'Found {len(results)} recordio files.')
    print(' '.join(results))
    print(f'Not found: {non_exist_files}')
