from datakit.utils.files import find_all_files, read_jsonl_file, dump_list_to_jsonl_file
from concurrent.futures import ThreadPoolExecutor, as_completed
# from datakit.api_call.gpt4_o import gpt4o_api 
import os
import random
import copy
import base64
import io
from typing import List

import numpy as np
from PIL import Image, ImageDraw


def decode_base64_image_to_np(base64_image: str) -> np.ndarray:
    """Decode a base64 string of an image to a numpy array.

    Args:
        base64_image (str): The base64 string of the image.

    Returns:
        np.ndarray: The numpy array of the image.
    """
    img_data = base64.b64decode(base64_image)
    img = Image.open(io.BytesIO(img_data))
    return np.array(img)

# complex results to cot input
# 带有是否为困难标签的text、easy、diff数据，输出仅仅包含diff text
root = "/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B"
root = "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/MATH/geoqa+/data/grammar_correct"
root = "/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results"
root = "/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_cot/geoqa+/data/grammar_correct_vlm_infer_Qwen72B_complexity_vllm_pe"
save_root = "/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_few_shot"
jsonls = find_all_files(root, '.jsonl')

num = 0
data = read_jsonl_file("/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results/geoqa+.jsonl")
for item in data:
    base64_image_dict = item['base64_image']
    for image_name, base64_image in base64_image_dict.items():
        np_image = decode_base64_image_to_np(base64_image)
        height, width = np_image.shape[:2]
        if not (height > 28 and width > 28):
            print(height, width)
            num += 1

print(num)

breakpoint()
"""
{'base64_image': {'/mnt/cephfs/bensenliu/dataset/mm/geoqa+/images/5238.png': 'iVBORw0KGgoAAAANSUhEUgAAAJ0AAAATCAAAAACv91CiAAACC0lEQVR4nMWVv27TYBTFf/cTU1+ApUVQhFQYgsQL0IEhRbCxJK/AlqELUtV2Y4+YUwneAFIhgdp06AN0qDtVqhIkWGBpWBLfw2AntQK2Q2PBlezvj4/OOfd+17aJ/xKyMsAO29yYrLbZrkB0TpY5YIZBWNTPtaKscABoxt0/O+V5ha7c2ZwpLSz5FzHtOw6qYD+ct3vLYYdk3VUQVmX51qsmrDSEVfnOVpynGQop59HZwvzGntnG+FUxT9PMbOVnCVfTzGzfkCRpVDuVXMl1zejTkvq0VUzTiKSt5csiJpcaXfWIUnc7z4fXt5XEqNaSXFvdIlFp9EKSGq0SskdD9YkCyP3twy8LHSrAu++7YKyu5iIM4PgWwL3TXJQAjteX2GytIbku2v364qVrZ2qUf7JJbTsleh0gkgIYb17yYNHSfT25M0m96K8zfv8Y4LxE72Okzv0zArD32laCO+4O7kAynQ7TWXrHHf+N0TL3/CTWl4DB7pNchIDBjzWatU8EGFzG6t0mEBTwEBw8ZAfAQ2CydjzgYcbezdo5wGC/QBQ4WAXYrG/kujPQ52fw7eQuqF9XrE47lmJJcTxOZ+N0wyUplny6TgDjqy5LW6UrHV71k8+MaXdGUp+nf3iexTUijWrLQ7EFZ3EDWnEirtSMYs1sZLfjyVYmjoBW0ZfO1Utq8yEfI0k9AOru+gWs8kbZVSKyzgAAAABJRU5ErkJggg=='}, 'conversations': [{'role': 'user', 'text': "已知:如图,点C是线段AB的中点,点D是线段BC的中点,AB=20cm,那么线段AD等于()\nA. 10cm\nB. 5cm\nC. 15cm\nD. 16cm\nAnswer with the option's letter from the given choices directly."}, {'role': 'assistant', 'text': 'C'}], 'id': '/mnt/cephfs/bensenliu/dataset/mm/geoqa+/images/5238.png', 'difficult_conversations': [{'role': 'user', 'text': "已知:如图,点C是线段AB的中点,点D是线段BC的中点,AB=20cm,那么线段AD等于()\nA. 10cm\nB. 5cm\nC. 15cm\nD. 16cm\nAnswer with the option's letter from the given choices directly."}, {'role': 'assistant', 'text': 'C', 'qwen2.5vl-72b': None}], 'easy_conversations': [{'role': 'user', 'text': "已知:如图,点C是线段AB的中点,点D是线段BC的中点,AB=20cm,那么线段AD等于()\nA. 10cm\nB. 5cm\nC. 15cm\nD. 16cm\nAnswer with the option's letter from the given choices directly."}, {'role': 'assistant', 'text': 'C', 'qwen2.5vl-3b': None}], 'complexity': 'hard'}
"""

num = 0
for _idx, jsonl in enumerate(jsonls):
    # if "geoqa+" not in jsonl:
    #     continue
    print(jsonl)
    # 构造fewshot数据
    data = read_jsonl_file(jsonl)
    num += len(data)
    for item in data:
        base64_image_dict = item['base64_image']
        for image_name, base64_image in base64_image_dict.items():
            np_image = decode_base64_image_to_np(base64_image)
            height, width = np_image.shape[:2]
            if image_name == "/mnt/cephfs/bensenliu/dataset/mm/geoqa+/images/5238.png":
                print(height, width)
                breakpoint()
            if not (height > 28 and width > 28):
                print(height, width)
                breakpoint()
    
    print(num)

    continue

    easy_data = [{'id': item['id'], 'base64_image': item['base64_image'], 'conversations': item['conversations']} for item in data if item['complexity'] == 'easy']
    hard_data = [{'id': item['id'], 'base64_image': item['base64_image'], 'conversations': item['conversations']} for item in data if item['complexity'] == 'hard']
    
    print("easy_data")
    for item in easy_data:
        base64_image_dict = item['base64_image']
        for image_name, base64_image in base64_image_dict.items():
            np_image = decode_base64_image_to_np(base64_image)
            height, width = np_image.shape[:2]
            assert height > 28 and width > 28, f"image {image_name} height {height} width {width} is not valid"
    
    print("hard_data")
    for item in hard_data:
        base64_image_dict = item['base64_image']
        for image_name, base64_image in base64_image_dict.items():
            np_image = decode_base64_image_to_np(base64_image)
            height, width = np_image.shape[:2]
            assert height > 28 and width > 28, f"image {image_name} height {height} width {width} is not valid"
    
    continue
    
    hard_data_random = copy.deepcopy(hard_data)
    for item in hard_data:
        few_shot_samples = random.sample(easy_data + hard_data_random, min(3, len(easy_data + hard_data_random)))
        new_conversations = []
        cur_b64 = item['base64_image']
        item['base64_image'] = {}
        for sample in few_shot_samples:
            item['base64_image'].update(sample['base64_image'])
            assert len(sample['conversations']) == 2
            sample_0_text = sample['conversations'][0]['text']
            sample_0_text += " " + sample['conversations'][1]['text']
            new_conversations.append({
                'role': 'user',
                'text': sample_0_text,
            })
        item['base64_image'].update(cur_b64)
        assert len(item['conversations']) == 2
        item['conversations'] = new_conversations + item['conversations']

    merged_data = easy_data + hard_data

    # 等分八份后删除原文件
    folder_name = jsonl.split("/")[-1].replace(".jsonl", "")
    os.makedirs(os.path.join(save_root, folder_name, "data", "grammar_correct"), exist_ok=True)
    random.shuffle(merged_data)  # 在分割前打乱顺序

    total = len(merged_data)
    n_split = 8
    chunk_size = (total + n_split - 1) // n_split  # 向上取整
    base_dir = os.path.join(save_root, folder_name, "data", "grammar_correct")

    for i in range(n_split):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total)
        chunk_lines = merged_data[start:end]
        out_path = os.path.join(base_dir, f"{i:02d}.jsonl")
        dump_list_to_jsonl_file(out_path, chunk_lines)

    print(f"save to {os.path.join(save_root, folder_name, 'data', 'grammar_correct')}")
