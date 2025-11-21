import argparse
import base64
import json
import os
import time
from datetime import datetime

from openai import OpenAI
from sglang.utils import wait_for_server
from tqdm import tqdm

from datakit.utils.distributed import (
    barrier_all_processes,
    dist_split_files,
    get_distributed_env,
    delete_folder
)
from datakit.utils.files import dump_list_to_jsonl_file, find_all_files, read_jsonl_file
from datakit.utils.image import save_base64_image
from datakit.utils.mp import multi_process_with_append

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset-file", type=str, required=True, help="path to the dataset file"
)
parser.add_argument(
    "--model-name", type=str, default="qwen2.5vl-7b", help="name of the model"
)
parser.add_argument("--task-name", type=str, required=True, help="name of the task")
parser.add_argument(
    "--num-workers",
    type=int,
    default=128,
    help="number of workers for multi-processing",
)
parser.add_argument(
    "--model-path", type=str, required=True, help="path of the model"
)
parser.add_argument(
    "--output-path-suffix", type=str, required=True, help="suffix of output path"
)
parser.add_argument(
    "--log-path", type=str, required=True, help="log path"
)


print("Waiting post in...")
wait_for_server("http://127.0.0.1:8081/")
print("Post prepared!")

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:8081/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
print("Client prepared!")


def query_vllm(prompt: str, image: base64, model_path: str) -> str:
    image_id = str(abs(hash(image)))
    try:
        cur_time = str(datetime.now())
        image_id = image_id + cur_time
        image_path = os.path.join("/tmp", image_id + ".jpg")
        save_base64_image(image, image_path)
        chat_response = client.chat.completions.create(
            model=model_path,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{prompt}"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"file://{image_path}"},
                        },
                    ],
                },
            ],
            max_tokens=2048,
            temperature=0.0,
        )
        return chat_response.choices[0].message.content

    except Exception as e:
        print(e)
        return None


def query_single_item(item: dict, model_name: str, model_path: str) -> dict:
    """Query a single item using the model.

    Args:
        item (dict): a single item in the dataset.
        model_name (str): the name of the model.

    Returns:
        dict: the query result of the item.
    """
    base64_image = item["base64_image"][list(item["base64_image"].keys())[0]]
    conversations = item["conversations"]
    for i in range(len(conversations) // 2):
        prompt = conversations[2 * i]["text"]
        response = query_vllm(prompt, base64_image, model_path)
        conversations[2 * i + 1][model_name] = response
    item["conversations"] = conversations
    return item


def query_single_item_wrapper(args: list):
    """Wrapper function for multi-processing.

    Args:
        args (list): a list of arguments for the query_single_item function.

    Returns:
        dict: the query result of the item.
    """
    item, model_name, model_path = args
    try:
        item = query_single_item(item, model_name, model_path)
    except Exception as e:
        print(e)
        item = None
    return item


if __name__ == "__main__":
    args = parser.parse_args()
    delete_folder(root=args.log_path, task_name=args.task_name)
    with open(args.dataset_file, "r") as f:
        datasets = f.readlines()
    datasets = [dataset.strip() for dataset in datasets]
    _, rank, _ = get_distributed_env()
    for dataset in tqdm(datasets):
        output_dataset = dataset + args.output_path_suffix
        os.makedirs(output_dataset, exist_ok=True)
        all_jsonls = find_all_files(dataset, "jsonl")
        all_data = []
        for jsonl in all_jsonls:
            data = read_jsonl_file(jsonl)
            data = [[item, args.model_name, args.model_path] for item in data]
            all_data.extend(data)
        data_cur_rank = dist_split_files(all_data)

        results = multi_process_with_append(
            query_single_item_wrapper,
            data_cur_rank,
            min(args.num_workers, len(data_cur_rank)),
        )

        output_jsonl = os.path.join(output_dataset, f"{rank:02d}.jsonl")
        if len(results) > 0:
            dump_list_to_jsonl_file(output_jsonl, results)
    barrier_all_processes(task_name=args.task_name, root=args.log_path)
    print("All tasks are finished.")
