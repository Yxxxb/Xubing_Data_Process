from datasets import load_dataset
from tqdm import tqdm
import random
from datakit.utils.image import encode_pil_to_base64_image
from datakit.utils.files import dump_list_to_jsonl_file



if __name__ == '__main__':
    data_root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/food101'
    output_file = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/food101/food101.jsonl'
    dataset = load_dataset(data_root, split='train')
    dataset_dict = {}
    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        pil_image = item['image']
        base64_image = encode_pil_to_base64_image(pil_image)
        label = item['label']
        if label not in dataset_dict:
            dataset_dict[label] = []
        dataset_dict[label].append(base64_image)
    final_data = []
    for _, values in dataset_dict.items():
        sample_values = random.sample(values, min(100, len(values)))
        final_data.extend(sample_values)
    dump_list_to_jsonl_file(output_file, final_data)
    