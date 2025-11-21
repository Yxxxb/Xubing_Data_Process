from datakit.utils.files import read_mmq_index
from data_distribution import data_distribution
import json







if __name__ == '__main__':
    reader = read_mmq_index('/mnt/cephfs/bensenliu/wfs/datasets/mm/sft/composition/202412/20241219/e1_sft_index.recordio')
    output_file = 'points-1.2/data/sft_pack/data_distribution_20241219e1.json'
    files = reader.header.filenames
    file_names = [file.split('/')[-4] for file in files]
    print(file_names)
    count = 0
    new_data_distribution = {}
    for key in data_distribution:
        new_data_distribution[key] = []
        for dataset in data_distribution[key]:
            if dataset in file_names:
                count += 1
                new_data_distribution[key].append(dataset)
    with open(output_file, 'w') as f:
        json.dump(new_data_distribution, f, indent=4)
    print(f'Total number of files: {len(files)}')
    


    