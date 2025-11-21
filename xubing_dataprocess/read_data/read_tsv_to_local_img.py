import pandas as pd
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import os
import io

from datakit.utils.files import find_all_files, read_jsonl_file
from datakit.utils.image import decode_base64_image_to_pil, draw_bounding_box, save_base64_image

import os.path as osp
import base64
from PIL import Image

def decode_base64_to_image(base64_string, target_size=-1):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    if image.mode in ('RGBA', 'P', 'LA'):
        image = image.convert('RGB')
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image

def decode_base64_to_image_file(base64_string, image_path, target_size=-1):
    image = decode_base64_to_image(base64_string, target_size=target_size)
    base_dir = osp.dirname(image_path)
    if not osp.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    image.save(image_path)

df = pd.read_csv('/mnt/cephfs/bensenliu/wfs/mm_datasets/eval/LiveVQA_old.tsv', sep='\t')
op_root = "/mnt/cephfs/xubingye/rubbish/imgs"

# valid_rows = []
# error_c = 0
# right_c = 0

# for idx, row in df.iterrows():
#     try:
#         decode_base64_to_image_file(row['image'], "/mnt/cephfs/xubingye/rubbish/imgs/rbsh.jpg")
#         valid_rows.append(row)
#         right_c += 1
#     except Exception:
#         error_c += 1
#         continue

# new_df = pd.DataFrame(valid_rows)

# # 保存新的tsv文件
# print(f"right_c: {right_c}, error_c: {error_c}")
# new_tsv_path = osp.join(op_root, "valid_data.tsv")
# new_df.to_csv('/mnt/cephfs/bensenliu/wfs/mm_datasets/eval/LiveVQA_fixed.tsv', sep='\t', index=False)


import hashlib

def calculate_md5(file_path):
    # 创建一个新的 md5 哈希对象
    md5 = hashlib.md5()

    try:
        with open(file_path, 'rb') as f:
            # 分块读取文件以处理大文件
            while chunk := f.read(8192):
                md5.update(chunk)
        
        # 返回校验和的十六进制表示
        return md5.hexdigest()
    
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
        return None

file_path = '/mnt/cephfs/bensenliu/wfs/mm_datasets/eval/LiveVQA_fixed.tsv'
md5_checksum = calculate_md5(file_path)

if md5_checksum is not None:
    print(f"MD5 checksum of the file is: {md5_checksum}")