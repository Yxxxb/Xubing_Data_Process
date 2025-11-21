import pandas as pd
import re

from datakit.utils.files import (dump_list_to_jsonl_file, find_all_files, mem_efficient_read_jsonl_file,
                                 read_jsonl_file)
from datakit.utils.image import decode_base64_image_to_pil
import os
import random
import base64
from io import BytesIO
from typing import Dict, Any
import requests


def call_crawl4ai(url: str) -> Dict[str, Any]:
    """Call the crawl4ai endpoint and return the result.

    Args:
        url (str): The url to crawl.
    Returns:
        Dict[str, Any]: The crawl4ai result.
    """
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    data = {'url': url}
    response = requests.post(
        'http://mmsprllmappsvr-tke.polaris:8880/crawl4ai',  # noqa
        headers=headers,
        json=data,
        timeout=60)
    if response.status_code == 200:
        response = response.json()
        return response['markdown']
    else:
        response = requests.post(
            'http://mmsprllmappsvr-tke-inner.polaris:8880/crawl4ai',  # noqa
            headers=headers,
            json=data,
            timeout=60)
        if response.status_code == 200:
            response = response.json()
            return response['markdown']
        else:
            raise Exception('Failed to call crawl4ai.')
            

def call_crawl4ai_with_retry(url: str, max_retry: int = 3, retry: int = 0) -> str:
    """Call the crawl4ai endpoint with retry.

    Args:
        url (str): The url to crawl.
        max_retry (int, optional): The maximum number of retries.
            Defaults to 3.
        retry (int, optional): The current retry count.
            Defaults to 0.

    Returns:
        str: The crawled content.
    """
    if retry >= max_retry:
        return None
    try:
        result = call_crawl4ai(url)
        return result
    except Exception as e:
        print(f'An error occurred: {e}')
        result = call_crawl4ai_with_retry(url, max_retry, retry + 1)
        return result


aa = read_jsonl_file('/mnt/cephfs/xubingye/wfs/datasets/rl-category/arxiv_search/arxiv_search.jsonl')

# 获取指定路径下所有的png文件
image_dir = '/mnt/cephfs/xubingye/tsp/ft_local/arxiv'
png_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]

if not png_files:
    raise FileNotFoundError("未在指定路径下找到PNG文件")

# 遍历aa中的每个sample
for sample in aa:
    # 随机选择一个png文件
    random_png = random.choice(png_files)
    try:
        # 读取图片并编码为base64
        with open(random_png, 'rb') as img_file:
            img_data = img_file.read()
            b64_encoded = base64.b64encode(img_data).decode('utf-8')
            # 将base64编码数据添加到sample中
            sample['b64_image'] = b64_encoded
    except Exception as e:
        print(f"处理图片 {random_png} 时出错: {e}")


# 将处理后的结果保存为jsonl文件
output_path = '/mnt/cephfs/xubingye/wfs/datasets/rl-category/arxiv_search/arxiv_search_b64.jsonl'
dump_list_to_jsonl_file(output_path, aa)

"""
React
1. 重点还是long memory，可能基于语言、embedding、自我提升
关键点：怎么定义好的环境
2. reward很关键，基于规则的，理想的就应该是这样，白盒。基于过程和黑盒、人和机器的偏好，一定会引起hacking
内在的reward，这是所有innovator最重要的事情，获得诺贝尔学降前怎么给自己reward
内生的奖励系统。创新者在没有外在的激励下做很多事情。
3. multi-agent，agent之间怎么scale

context很关键，人可能数学推理能力不如o3，但是一个凡人入职一个公司七天，他会有一个context，这个东西无法用自然语言描述。
所以o3替代不了人。人之所以能generalize是因为他能够推理。
第一性原理。
"""