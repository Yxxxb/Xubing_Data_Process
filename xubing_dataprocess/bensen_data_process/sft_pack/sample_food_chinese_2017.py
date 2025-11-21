from datasets import load_dataset
from tqdm import tqdm
import random
from datakit.utils.image import encode_bytes_to_base64_image
from datakit.utils.files import dump_list_to_jsonl_file


if __name__ == '__main__':
    data_root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/food_chinese_2017'
    output_file = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/food_chinese_2017/food_chinese_2017.jsonl'
    dataset = load_dataset(data_root, split='train')
    dataset_dict = {}
    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        bytes_image = item['image']['bytes']
        base64_image = encode_bytes_to_base64_image(bytes_image)
        label = item['label']
        if label not in dataset_dict:
            dataset_dict[label] = []
        dataset_dict[label].append(base64_image)
    final_data = []
    for _, values in dataset_dict.items():
        sample_values = random.sample(values, min(50, len(values)))
        final_data.extend(sample_values)
    dump_list_to_jsonl_file(output_file, final_data)
    