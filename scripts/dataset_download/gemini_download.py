import argparse
import json
import logging
import os
import time
from multiprocessing import Pool
from pathlib import Path

import requests
from datazoo.utils.generic import HadoopClient, load_text  # noqa
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    '--url_file',
    type=Path,
    required=True,
    help='Path to the file containing the download links and sizes',
)
parser.add_argument(
    '--save_path',
    type=Path,
    required=True,
    help='Path to the save dir, start with [[username]], \
        e.g. ziyzhuang/MINT-1T/',
)
parser.add_argument(
    '--link_prefix',
    type=str,
    required=False,
    help='Link prefix to strip, default as \
        https://huggingface.co/datasets/[[DATASET_NAME]]/',
)
args = parser.parse_args()

os.environ['JAVA_HOME'] = '/data/offline/jdk'
os.environ['http_proxy'] = 'http://hk-mmhttpproxy.woa.com:11113'
os.environ['https_proxy'] = 'http://hk-mmhttpproxy.woa.com:11113'
os.environ['no_proxy'] = 'wandb.dev.woa.com'

CPU_CORE_NUM = int(float((os.environ.get('CPU_CORE_NUM', 2))))
TASK_INDEX = int(os.environ.get('TASK_INDEX'))
TASK_CNT = int(os.environ.get('TASK_CNT'))
WFS_ROOT = 'wfs://mmnanjingwfssh:17024/project_pr-nlp-large_pretrain/'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

hadoop_client = HadoopClient(
    '/data/offline/hadoop/bin/hadoop',
    os.environ['HADOOP_JOB_UGI'],
)


def filter_done(url):
    size, download_url = url
    src_path = download_url.replace(args.link_prefix, SAVE_PATH)
    if size == hadoop_client.get_filesize(src_path):
        return None
    return url


def download_file(url, save_path):
    headers = None
    if AUTH_TOKEN is not None:
        headers = {
            'Authorization': f'Bearer {AUTH_TOKEN}',
        }
    response = requests.get(url, headers=headers, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(url, 'download success')
        logger.info(f'{url} download success to {save_path}')
    else:
        logger.warning(
            f'{url} download fail, status code: {response.status_code}')


def download(size_url_item):
    filesize, url = size_url_item
    filename = os.path.basename(url) + '-' + str(time.time())
    src_path = url.replace(args.link_prefix, SAVE_PATH)
    download_file(url, filename)
    if os.path.exists(filename) and os.path.getsize(filename) == filesize:
        hadoop_client.put(filename, src_path, remove=True)
        if hadoop_client.get_filesize(src_path) == filesize:
            logger.info(f'{url} hdfs put success to {src_path}')
            return None

    raise Exception(f'{url} hdfs put fail')


def download_url(size_url):
    MAX_RETRY = 3
    for retry in range(MAX_RETRY):
        try:
            download(size_url)
            return
        except Exception as e:
            logger.warning(f'{size_url} download error: {e}, retry {retry}')
            time.sleep(10)
    logger.error(f'{size_url} download fail')


if __name__ == '__main__':
    # !===================================
    # !## MODIFY HERE FOR YOUR OWN CODE ##
    SAVE_PATH = os.path.join(WFS_ROOT, args.save_path)
    if not SAVE_PATH.endswith('/'):
        SAVE_PATH += '/'
    AUTH_TOKEN = os.environ['HF_AUTH_TOKEN']
    with open(args.url_file) as f:
        size_url_list = json.load(f)
    # !===================================

    if args.link_prefix is None:
        args.link_prefix = size_url_list[0][1].split('resolve/main/')[0]

    size_url_list = size_url_list[TASK_INDEX:len(size_url_list):TASK_CNT]
    unfinished = []
    with Pool(processes=min(len(size_url_list), CPU_CORE_NUM)) as pool:
        for size_url in tqdm(
                pool.imap_unordered(filter_done, size_url_list),
                desc='Filtering download success files',
        ):
            if size_url:
                unfinished.append(size_url)

    logger.info(f'{TASK_INDEX}-{TASK_INDEX} total: {len(size_url_list)},'
                f' remain: {len(unfinished)}')
    if len(unfinished) > 0:
        for size_url in tqdm(unfinished, desc='Downloading files'):
            download_url(size_url)

    print('Done')
    if TASK_INDEX == 0:
        print('Master sleep...')
        time.sleep(3600)
