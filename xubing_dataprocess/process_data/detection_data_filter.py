import os
import sys
import argparse
# from transformers import AutoTokenizer
# from datakit.utils.files import read_mmq_index, read_mmq_recordio
from datakit.utils.files import dump_list_to_jsonl_file, find_all_files, read_jsonl_file
from datakit.utils.image import decode_base64_image_to_pil, draw_bounding_box

import json
import random
import re

data = read_jsonl_file("/mnt/cephfs/bensenliu/wfs/vlmdatasets/pt/laion5b-en/data/mmgrounding_dino_detection/0_000000.jsonl")
images = read_jsonl_file("/mnt/cephfs/bensenliu/wfs/vlmdatasets/pt/laion5b-en/data/0/000000.jsonl")
count_original_gt5 = 0
count_deduplicated_gt5 = 0
count_max_freq_gt5 = 0
save_bbox_dir = "/mnt/cephfs/bensenliu/wfs/vlmdatasets/pt/laion5b-en/data/mmgrounding_dino_detection_bbox"
    
for item, item_image in zip(data, images):
    assert 'object_list' in item['mm_grounding_dino']
    object_list = item['mm_grounding_dino']['object_list']

    # 原始长度大于10
    if len(object_list) > 10:
        count_original_gt5 += 1

    # 统计每个对象的频次
    freq_count = {}
    for obj in object_list:
        freq_count[obj] = freq_count.get(obj, 0) + 1
    
    # 找到最高频次
    max_freq = max(freq_count.values()) if freq_count else 0
    
    # 检查最高频次是否大于5
    if max_freq > 10:
        count_max_freq_gt5 += 1
    
    # 去重后长度大于10
    unique_objects = list(set(object_list))
    if len(unique_objects) > 10:
        count_deduplicated_gt5 += 1
        bboxs = item['mm_grounding_dino']['bbox']
        image = decode_base64_image_to_pil(item_image['base64_image'])
        assert len(object_list) == len(bboxs), "Object list and bbox list length mismatch"
        for idx, bbox in enumerate(bboxs):
            bbox[0] = bbox[0] / image.size[0]
            bbox[1] = bbox[1] / image.size[1]
            bbox[2] = bbox[2] / image.size[0]
            bbox[3] = bbox[3] / image.size[1]
        image_bbox = draw_bounding_box(image, bbox)
        image_bbox.save(os.path.join(save_bbox_dir, f"{item['id']}_{obj}.png"))





print(f"原始object_list长度大于5的数量: {count_original_gt5}")
print(f"去重后object_list长度大于5的数量: {count_deduplicated_gt5}")
print(f"object_list中最高频次大于5的数量: {count_max_freq_gt5}")
print(len(data))

breakpoint()
