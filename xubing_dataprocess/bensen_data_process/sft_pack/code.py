from data_distribution_stable_version import data_distribution
import os



root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category'
dataset_paths = []
for category, dataset_list in data_distribution.items():
    for dataset in dataset_list:
        dataset_path = os.path.join(root, category, dataset,
                                    f'data/grammar_correct/{dataset}.jsonl')
        if os.path.exists(dataset_path):
            dataset_paths.append(dataset_path)
print('\n'.join(dataset_paths))
