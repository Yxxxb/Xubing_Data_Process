import requests
import json
import os
from tqdm import tqdm
from data_distribution import data_distribution
from datakit.utils.distributed import dist_split_files
from datakit.utils.files import find_all_files, read_jsonl_file, dump_list_to_jsonl_file
from datakit.utils.distributed import dist_split_files
from datakit.utils.mp import multi_process_with_append

url = "http://127.0.0.1:8081/v1/chat/completions"

with open('points-1.2/prompts/reasoning_general_qa.txt') as f:
    PROMPT = f.read()

def obtain_results(item):
    base64_image_key = list(item['base64_image'].keys())[0]
    base64_image = item['base64_image'][base64_image_key]
    data = {
        "model": "Qwen/Qwen2-VL-7B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    }
                ],
            }
        ],
        "max_tokens": 4096,
    }
    response = requests.post(url, json=data)
    response = json.loads(response.text)
    response = response['choices'][0]['message']['content']
    conversations = [
        {
            "role": "user",
            "text": 'Generated response'
        },
        {
            "role": "assistant",
            "text": response
        }
    ]
    item['conversations'] = conversations
    return  item


if __name__ == '__main__':
    ratio = 0.2
    SUBSET = 'grammar_correct'
    root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category'
    include_datasets = [
        'sharegpt4v_llava158k',
        'sharegpt4v_qa_cn'
    ]
    points = []
    for category, dataset_names in data_distribution.items():
        for dataset_name in dataset_names:
            if dataset_name not in include_datasets:
                continue
            points.append(f'{category}/{dataset_name}')
    points_cur_rank = dist_split_files(points)
    points_cur_rank = [os.path.join(root, point) for point in points_cur_rank]
    for cur_dataset in tqdm(points_cur_rank):
        cur_folder = os.path.join(cur_dataset, 'data', SUBSET)
        output_folder = os.path.join(cur_dataset, 'data', SUBSET+f'qwen2-vl-reasoning-{int(ratio*100)}-per')
        output_path = os.path.join(output_folder, f'{cur_dataset.split("/")[-1]}.jsonl')
        os.makedirs(output_folder, exist_ok=True)
        cur_file = find_all_files(cur_folder, 'jsonl')[0]
        data = read_jsonl_file(cur_file)
        data = data[-int(len(data)*ratio):]
        results = []
        for i in range(0, len(data), 1000):
            cur_data = data[i:i+1000]
            cur_results = multi_process_with_append(obtain_results, cur_data, num_workers=10)
            results.extend(cur_results)
            dump_list_to_jsonl_file(output_path, results)
    print('Done')
        
