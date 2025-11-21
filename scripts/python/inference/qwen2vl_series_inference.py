import argparse
import base64
import json
import os
from datetime import datetime

import requests
from sglang.utils import wait_for_server
from tqdm import tqdm

from datakit.utils.distributed import (barrier_all_processes, dist_split_files,
                                       get_distributed_env)
from datakit.utils.files import (dump_list_to_jsonl_file, find_all_files,
                                 read_jsonl_file)
from datakit.utils.image import save_base64_image
from datakit.utils.mp import multi_process_with_append

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-file',
                    type=str,
                    required=True,
                    help='path to the dataset file')
parser.add_argument('--model-name',
                    type=str,
                    default='qwen2.5vl-7b',
                    help='name of the model')
parser.add_argument('--task-name',
                    type=str,
                    required=True,
                    help='name of the task')
parser.add_argument('--num-workers',
                    type=int,
                    default=128,
                    help='number of workers for multi-processing')

url = 'http://127.0.0.1:8081/v1/chat/completions'


def query_sglang(prompt: str, image: base64) -> str:
    image_id = str(abs(hash(image)))
    try:
        cur_time = str(datetime.now())
        image_id = image_id + cur_time
        image_path = os.path.join('/tmp', image_id + '.jpg')
        save_base64_image(image, image_path)
        data = {
            'model':
            'Qwen/Qwen2.5-VL-7B-Instruct',
            'messages': [{
                'role':
                'user',
                'content': [
                    {
                        'type': 'text',
                        'text': f'{prompt}'
                    },
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': image_path
                        },
                    },
                ],
            }],
            'max_new_tokens':
            2048,
            'temperature':
            0.0,
            'do_sample':
            False,
        }
        response = requests.post(url, json=data)
        os.remove(image_path)
        response = json.loads(response.text)
        response = response['choices'][0]['message']['content']
        return response
    except Exception as e:
        print(e)
        return None


def query_single_item(item: dict, model_name: str) -> dict:
    """Query a single item using the model.

    Args:
        item (dict): a single item in the dataset.
        model_name (str): the name of the model.

    Returns:
        dict: the query result of the item.
    """
    base64_image = item['base64_image'][list(item['base64_image'].keys())[0]]
    conversations = item['conversations']
    for i in range(len(conversations) // 2):
        prompt = conversations[2 * i]['text']
        response = query_sglang(prompt, base64_image)
        conversations[2 * i + 1][model_name] = response
    item['conversations'] = conversations
    return item


def query_single_item_wrapper(args: list):
    """Wrapper function for multi-processing.

    Args:
        args (list): a list of arguments for the query_single_item function.

    Returns:
        dict: the query result of the item.
    """
    item, model_name = args
    try:
        item = query_single_item(item, model_name)
    except Exception as e:
        print(e)
        item = None
    return item


if __name__ == '__main__':
    wait_for_server('http://127.0.0.1:8081')
    args = parser.parse_args()
    with open(args.dataset_file, 'r') as f:
        datasets = f.readlines()
    datasets = [dataset.strip() for dataset in datasets]
    _, rank, _ = get_distributed_env()
    for dataset in tqdm(datasets):
        output_dataset = dataset + '_vlm_infer'
        os.makedirs(output_dataset, exist_ok=True)
        all_jsonls = find_all_files(dataset, 'jsonl')
        all_data = []
        for jsonl in all_jsonls:
            data = read_jsonl_file(jsonl)
            data = [[item, args.model_name] for item in data]
            all_data.extend(data)
        data_cur_rank = dist_split_files(all_data)
        results = multi_process_with_append(
            query_single_item_wrapper, data_cur_rank,
            min(args.num_workers, len(data_cur_rank)))
        output_jsonl = os.path.join(output_dataset, f'{rank:02d}.jsonl')
        if len(results) > 0:
            dump_list_to_jsonl_file(output_jsonl, results)
    barrier_all_processes(args.task_name)
    print('All tasks are finished.')
