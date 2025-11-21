import argparse
import base64
import json
import os
import time
from datetime import datetime
from tqdm import tqdm

from datakit.utils.distributed import (
    barrier_all_processes,
    dist_split_files,
    get_distributed_env,
    delete_folder
)
from datakit.utils.files import dump_list_to_jsonl_file, find_all_files, read_jsonl_file, mem_efficient_read_jsonl_file
from datakit.utils.image import save_base64_image
from datakit.utils.mp import multi_process_with_append

from typing import List, Tuple
import numpy as np
from io import BytesIO
from PIL import Image
from openai import OpenAI
from sglang.utils import wait_for_server

import re

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num-workers",
    type=int,
    default=64,
    help="number of workers for multi-processing",
)

parser.add_argument(
    "--root-path", type=str, default="/mnt/cephfs/haichengwang/code/points_r1/tool_sft_correct", help="path of the root"
)
parser.add_argument(
    "--partition", type=str, default=None, help="sub partitions to be processed, e.g. 0-10"
)
parser.add_argument(
    "--task-name", type=str, default="cold_start", help="sub partitions to be processed, e.g. 0-10"
)
parser.add_argument(
    "--barrier-path", type=str, default=None, help="path of the barrier file"
)

parser.add_argument(
    "--local-rank", type=int, default=0, help="data parallel local rank"
)
parser.add_argument(
    "--post", type=str, default='32520', help="post port num"
)
client = None

# MODEL_PATH = "/mnt/cephfs/kamillewang/wfs/models/Qwen3-235B-A22B"
MODEL_PATH = "/mnt/cephfs/haichengwang/wfs/opensource/Qwen3-32B"
MAX_TOKENS = 1024 * 1
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
PROMPT = (
'You are an AI assistant who will help me determine whether the answer to a fill-in-the-blank or multiple-choice question matches the ground truth.'
'You will be given a question (which may have options), an answer, and a ground truth.'
'If it is a fill-in-the-blank question, you will directly determine whether the answer matches the ground truth. '
'If the answer and the ground truth have the same meaning but different formats, they can be considered a match.'
'If it is a multiple-choice question and the answer is the content of an option rather than the option itself, '
'you need to identify which option is most similar to the answer and then determine whether it matches the ground truth option.'
'If the answer and the ground truth match, output Yes. If there is a significant discrepancy between the answer and the ground truth, output No.'
'You should only output Yes or No. \n'
'Example 1: \n'
'Question: What is the main object? A. Plush bear B. Rabbit C. Cat D. Dog\n'
'Answer: A cute rabbit\nGround truth: B\nYour output: Yes\n'
'Example 2: \n'
'Question: What is the main object? A. Plush bear B. Rabbit C. Cat D. Dog\n'
'Answer: Spider\nGround truth: B\nYour output: No\n'
'Example 3: \n'
'Question: What is the product of 3 and 3.5?\n'
'Answer: 10.5\nGround truth: \(\frac{{21}}{{2}}\)Your output: Yes\n'
'Example 4: \n'
'Question: {question}?\nAnswer: {answer}\nGround truth: {ground_truth}\nYour output: '
)
format_prompt = " You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."


def seconds_to_hms(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

def has_repetition(text, min_repeat=6, ngram=2):
    """
    检查文本中是否有连续重复的n-gram（如单词、词组等）
    """
    words = text.strip().split()
    if len(words) < ngram * min_repeat:
        return False
    ngrams = [' '.join(words[i:i+ngram]) for i in range(len(words)-ngram+1)]
    count = 1
    for i in range(1, len(ngrams)):
        if ngrams[i] == ngrams[i-1]:
            count += 1
            if count >= min_repeat:
                return True
        else:
            count = 1
    return False

def vllm_batch_inference(frame_metadata):
    assert client is not None
    offline_roll_results = frame_metadata["offline_roll_results"]
    for idx, roll_item in enumerate(offline_roll_results):
        match = re.search(r'\\boxed{(.*)}', roll_item)
        if match:
            offline_roll_results[idx] = match.group(1)
    gt = frame_metadata['reward_model']['ground_truth']
    question = frame_metadata['prompt'][0]['content'].replace(format_prompt, '')

    match_num = 0
    frame_metadata["match_log"] = []
    for roll_item in offline_roll_results:
        if gt.strip().lower() == roll_item.strip().lower():
            match_num += 1
            frame_metadata["match_log"].append("Exactly match.")
        else:
            messages = []
            content = []
            input_text = PROMPT.format(question=question, answer=roll_item, ground_truth=gt)
            content.append({"type": "text", "text": input_text})
            messages.append([
                {"role": "user", "content": content},
            ])
            try:
                chat_response = client.chat.completions.create(
                        model=MODEL_PATH,
                        messages=messages[0],
                        max_tokens=MAX_TOKENS,
                        temperature = 0.7,
                        top_p = 0.8,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    )
                cur_reponse = chat_response.choices[0].message.content
                if 'Yes' in cur_reponse or 'yes' in cur_reponse:
                    match_num += 1
                frame_metadata["match_log"].append(cur_reponse)
            except Exception as e:
                print(f"Error occurred while processing: {e}")
                input_tokens = tokenizer.encode(input_text, truncation=True, max_length=36864, return_tensors="pt")
                messages[0][0]["content"][0]["text"] = tokenizer.decode(input_tokens[0], skip_special_tokens=True)
                try:
                    chat_response = client.chat.completions.create(
                            model=MODEL_PATH,
                            messages=messages[0],
                            max_tokens=MAX_TOKENS,
                            temperature = 0.7,
                            top_p = 0.8,
                            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                        )
                    cur_reponse = chat_response.choices[0].message.content
                    if 'Yes' in cur_reponse or 'yes' in cur_reponse:
                        match_num += 1
                    frame_metadata["match_log"].append(cur_reponse)
                except Exception as e:
                    print(f"Error occurred again while processing: {e}")
                    frame_metadata["match_log"].append("Error")
    frame_metadata["match_num"] = match_num
    return frame_metadata


def query_jsonl_text(jsonl_file, output_root):
    """Wrapper function for multi-processing with batch inference.

    Args:
        args (list): a list of arguments for the query_single_item function.

    Returns:
        dict: the query result of the item.
    """
    output_jsonl = output_root
    # output_jsonl = os.path.join(output_root, os.path.basename(jsonl_file))
    if os.path.exists(output_jsonl):
        print(f"{output_jsonl} already exists, skip...")
        return
    whole_data = read_jsonl_file(jsonl_file)
    saved_data_list = []
    batch_size = 16
    for i, start in enumerate(range(0, len(whole_data), batch_size)):
        batch = whole_data[start:start + batch_size]
        start_time = time.time()
        try:
            caption_list = multi_process_with_append(vllm_batch_inference, batch, num_workers=len(batch))
        except Exception as e:
            print(f"Error occurred while processing batch {i}: {e}")
            continue
        saved_data_list.extend(caption_list)
        print(f"---------------------------------------------\n")
        print(f"Time for batch {i}: {time.time() - start_time}")
    if len(saved_data_list) > 0:
        dump_list_to_jsonl_file(output_jsonl, saved_data_list)
        print(f"Save to {output_jsonl}...")
    else:
        print(f"{output_jsonl} is empty after request...")

if __name__ == "__main__":
    args = parser.parse_args()

    print("Waiting post in...")
    wait_for_server(f"http://127.0.0.1:{args.post}/")
    print("Post prepared!")

    # Set OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = f"http://127.0.0.1:{args.post}/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    print("Client prepared!")

    alert_root = "/mnt/cephfs/xubingye/envs"
    task_name = args.task_name + str(args.partition)

    jsonl_folder_path = "/mnt/cephfs/xubingye/wfs/datasets/rl-category/MMK12/data_verl_parquet_offline_roll"
    output_path = "/mnt/cephfs/xubingye/wfs/datasets/rl-category/MMK12/data_verl_parquet_offline_roll_filter_result_exactly_match"
    os.makedirs(output_path, exist_ok=True)
    all_jsonls = find_all_files(jsonl_folder_path, 'jsonl')
    all_output_paths = [i.replace('data_verl_parquet_offline_roll', 'data_verl_parquet_offline_roll_filter_result_exactly_match') for i in all_jsonls]

    delete_folder(root=alert_root, task_name=task_name)
    print("Begin to inference")
    assert args.partition is not None
    start, end = args.partition.split("-")
    all_jsonls = all_jsonls[int(start):int(end)]
    all_output_paths = all_output_paths[int(start):int(end)]

    if args.barrier_path is not None:
        remove_single_file(args.barrier_path)
        print(f"#### remove barrier file: {args.barrier_path}")
        
    print(f"##### partition: {args.partition} deal with files: {all_jsonls}...")

    for cur_path in all_output_paths:
        os.makedirs(os.path.dirname(cur_path), exist_ok=True)

    assert len(all_jsonls) == len(all_output_paths)
    for cur_jsonl, cur_output_path in tqdm(zip(all_jsonls, all_output_paths), total=len(all_jsonls)):
        query_jsonl_text(cur_jsonl, cur_output_path)
    print("All tasks are finished.")
