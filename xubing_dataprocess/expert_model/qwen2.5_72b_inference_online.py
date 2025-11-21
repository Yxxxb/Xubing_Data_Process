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
import cv2
import numpy as np
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from qwen_vl_utils import process_vision_info
from io import BytesIO
from PIL import Image
from openai import OpenAI
from sglang.utils import wait_for_server

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num-workers",
    type=int,
    default=64,
    help="number of workers for multi-processing",
)

parser.add_argument(
    "--root-path", type=str, default="/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category-video/CAPTION/Koala-36M/ensemble/fps1_maxframe300000-correct", help="path of the root"
)
parser.add_argument(
    "--partition", type=str, default=None, help="sub partitions to be processed, e.g. 0-10"
)
parser.add_argument(
    "--task-name", type=str, default="koala_qwen25_72b_ensemble", help="sub partitions to be processed, e.g. 0-10"
)

parser.add_argument(
    "--local-rank", type=int, default=0, help="data parallel local rank"
)

MODEL_PATH = "/mnt/cephfs/kamillewang/wfs/models/Qwen3-235B-A22B"
MAX_TOKENS = 3500
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
PROMPT = (
'Now I have a video that has been divided into multiple shots. Each shot is annotated with the following information:\n\n'
'1. Shot number, indicated by [Shot x]\n'
'2. The time interval of the shot in the video, indicated by [Time Stamp]\n'
'3. A description of the shot, indicated by [Shot Caption]\n'
'4. Text appearing in the shot, indicated by [Text in Shot]\n'
'5. Objects present in the shot, indicated by [Object in Shot]\n\n'
'I now ask you to complete the following tasks based on the given annotated text information:\n\n'
'1. Based on all the shot information, generate a very detailed description of the entire video.\n'
'2. The generated description should be logically coherent and linguistically smooth.\n'
'3. Do not include any fabricated or hallucinated content that does not appear in the provided shot information.\n'
'4. The detailed description should emphasize the temporal (time-based) structure (e.g. indicate time interval xx:xx:xx-yy:yy:yy, but not single timestamp xx:xx:xx) and progression of the video\n'
'5. Some shots are closely related and short, please merge consequent short shots (interval less than 15s) to make the description smoother.\n'
'6. Only return the generated detailed caption, not anything else (no thinking process, no comments, no summaries)\n\n'
'The given annotated text information for this video is as follows:\n\n'
)

# PROMPT = """
# Now I have a video that has been divided into multiple shots. Each shot is annotated with the following information:

# 1. Shot number, indicated by [Shot x]  
# 2. The time location of the shot in the video, indicated by [Time Stamp]  
# 3. A description of the shot, indicated by [Shot Caption]  
# 4. Text appearing in the shot, indicated by [Text in Shot]  
# 5. Objects present in the shot, indicated by [Object in Shot]  

# I now ask you to complete the following tasks:

# 1. Based on all the shot information, generate a detailed description of the entire video.  
# 2. The generated description should be logically coherent and linguistically smooth.  
# 3. Do not include any fabricated or hallucinated content that does not appear in the provided shot information.  
# 4. The detailed description should emphasize the temporal (time-based) structure and progression of the video.
# """

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

def get_concat_ocr_results_for_a_shot(item):
    ocr_list = item['ocr']
    ocr_result = []
    for ocr in ocr_list:
        if ocr[-1] != '' and not ocr[-1].startswith('The'):
            tmp_ocr_list = ocr[-1].split('\n')
            tmp_ocr_list = list(set(tmp_ocr_list))
            ocr[-1] = '\n'.join(tmp_ocr_list)
            ocr_result.append(ocr[-1])
    ocr_result = list(set(ocr_result))
    return '\n'.join(ocr_result)


def get_class_names_for_a_shot(item):
    detection_list = item['detection']
    detection_result_list = []
    for detection in detection_list:
        class_names = detection['class_name']
        detection_result_list.extend(class_names)
    detection_result_list = list(set(detection_result_list))
    return ' '.join(detection_result_list)

def reformat_data(item):
    """Reformat the data to be compatible with the vllm batch inference."""
    metadata_list = item['metadata']
    prompt = ''
    for i, metadata in enumerate(metadata_list):
        item_ocr_result = get_concat_ocr_results_for_a_shot(metadata)
        if has_repetition(item_ocr_result):
            item_ocr_result = ''
        item_detection_result = get_class_names_for_a_shot(metadata)
        video_caption = metadata['video_caption']
        time_stamps = metadata['timestamps']
        start_time = seconds_to_hms(time_stamps[0])
        end_time = seconds_to_hms(time_stamps[-1])
        prompt += f'[Shot {i+1}]\n'
        prompt += f'[Time Stamp]: {start_time} - {end_time}\n'    
        prompt += f'[Shot Caption]: {video_caption}\n'
        prompt += f'[Text in Shot]: {item_ocr_result}\n'
        prompt += f'[Object in Shot]: {item_detection_result}\n'
        if i != len(metadata_list) - 1:
            prompt += '\n ------------------------ \n\n'
    return prompt

def vllm_batch_inference(item):
    index, frame_metadata = item
    messages = []
    content = []
    input_text = f"{PROMPT + frame_metadata}"
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
        return (index, chat_response.choices[0].message.content)
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
            return (index, chat_response.choices[0].message.content + " Truncated...")
        except Exception as e:
            print(f"Error occurred again while processing: {e}")
            return (index, "")

def query_jsonl_video(jsonl_file):
    """Wrapper function for multi-processing with batch inference.

    Args:
        args (list): a list of arguments for the query_single_item function.

    Returns:
        dict: the query result of the item.
    """
    output_jsonl = os.path.join(output_root, os.path.basename(jsonl_file))
    output_jsonl = output_jsonl.replace(".jsonl", f"_filerange_0_30.jsonl")
    if os.path.exists(output_jsonl):
        return
    whole_data = read_jsonl_file(jsonl_file)
    saved_data_list = []
    batch_size = 400
    for i, start in enumerate(range(0, len(whole_data), batch_size)):
        batch = whole_data[start:start + batch_size]
        start_time = time.time()
        batch_samples = [(i, reformat_data(item)) for (i, item) in enumerate(batch)]
        batch_keys = [item['video_id'] for item in batch]
        try:
            caption_list = multi_process_with_append(vllm_batch_inference, batch_samples, num_workers=len(batch_samples))
            caption_list = [caption for (index, caption) in sorted(caption_list, key=lambda x: x[0])]
        except Exception as e:
            print(f"Error occurred while processing batch {i}: {e}")
            caption_list = ["" for _ in batch_samples]
        for caption, video_id in zip(caption_list, batch_keys):
            saved_data_list.append({'video_id': video_id, 'caption': caption})
            print(f"Video {video_id} caption: {caption}")
        print(f"---------------------------------------------\n")
        print(f"Time for batch {i}: {time.time() - start_time}")
    dump_list_to_jsonl_file(output_jsonl, saved_data_list)

if __name__ == "__main__":
    args = parser.parse_args()
    alert_root = "/mnt/cephfs/haichengwang/envs"
    task_name = args.task_name + str(args.partition) + "__" + str(args.local_rank)
    root = args.root_path
    output_root = os.path.join(root.replace("/ensemble", "/llm_merge"), "caption_qwen2_5_72b_ll_merge_20w")

    delete_folder(root=alert_root, task_name=task_name)
    _, rank, _ = get_distributed_env()
    os.makedirs(output_root, exist_ok=True)
    print("Begin to inference")
    all_jsonls = find_all_files(root, "jsonl")
    if args.partition is not None:
        start, end = args.partition.split("-")
        all_jsonls = all_jsonls[int(start):int(end)]
    unprocessed_jsons = []
    for jsonl_file in all_jsonls:
        output_jsonl = os.path.join(output_root, os.path.basename(jsonl_file))
        output_jsonl = output_jsonl.replace(".jsonl", f"_filerange_0_30.jsonl")
        if not os.path.exists(output_jsonl):
            unprocessed_jsons.append(jsonl_file)
    all_jsonls = unprocessed_jsons
    data_cur_rank = dist_split_files(all_jsonls)

    for data_cur in tqdm(data_cur_rank):
        query_jsonl_video(data_cur)
    with open(os.path.join(output_root, f"partition_{str(args.partition)}_local_rank_{str(args.local_rank)}_results.txt"), "w") as f:
        f.write("All tasks are finished.")
    print("All tasks are finished.")
