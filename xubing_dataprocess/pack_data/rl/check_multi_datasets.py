import pandas as pd
import os
import glob
import numpy as np
import re
from tqdm import tqdm
import datasets
from datasets import Dataset, concatenate_datasets
from datakit.utils.files import dump_list_to_jsonl_file, find_all_files, read_jsonl_file, mem_efficient_read_jsonl_file, read_parquet_file
import json

datasets_path = [
    '/mnt/cephfs/xubingye/wfs/datasets/rl-category/MMK12/data_verl_parquet',
    '/mnt/cephfs/xubingye/wfs/datasets/rl-category/CharXiv/data_verl_parquet_check',
    '/mnt/cephfs/xubingye/wfs/datasets/rl-category/ChartQAPro/data_verl_parquet_check',
    '/mnt/cephfs/xubingye/wfs/datasets/rl-category/ChartMuseum/data_verl_parquet_check',
    '/mnt/cephfs/xubingye/wfs/datasets/rl-category/VisuLogic/data_verl_parquet_check',
    '/mnt/cephfs/xubingye/wfs/datasets/rl-category/VisualPuzzles/data_verl_parquet_check',
    '/mnt/cephfs/xubingye/wfs/datasets/rl-category/SciFIBench/data_verl_parquet_check',
    # '/mnt/cephfs/xubingye/wfs/datasets/rl-category/EMMA/data_verl_parquet_check',
    # '/mnt/cephfs/xubingye/wfs/datasets/rl-category/zerobench/data_verl_parquet_check',
    '/mnt/cephfs/xubingye/wfs/datasets/rl-category/IQBench/data_verl_parquet_check',
    '/mnt/cephfs/xubingye/wfs/datasets/rl-category/OCR-Reasoning/data_verl_parquet_check',
]

datasets_path = [
    '/mnt/cephfs/xubingye/wfs/datasets/rl-category/MMK12/data_verl_parquet/',
    '/mnt/cephfs/xubingye/wfs/datasets/rl-category/CharXiv/data_verl_parquet_check/',
    '/mnt/cephfs/xubingye/wfs/datasets/rl-category/ChartQAPro/data_verl_parquet_check/',
    '/mnt/cephfs/xubingye/wfs/datasets/rl-category/ChartMuseum/data_verl_parquet_check/',
    '/mnt/cephfs/xubingye/wfs/datasets/rl-category/VisuLogic/data_verl_parquet_check/',
    '/mnt/cephfs/xubingye/wfs/datasets/rl-category/VisualPuzzles/data_verl_parquet_check/',
    # '/mnt/cephfs/xubingye/wfs/datasets/rl-category/SciFIBench/data_verl_parquet_check/',
    # '/mnt/cephfs/xubingye/wfs/datasets/rl-category/EMMA/data_verl_parquet_check/',
    # '/mnt/cephfs/xubingye/wfs/datasets/rl-category/zerobench/data_verl_parquet_check/',
    '/mnt/cephfs/xubingye/wfs/datasets/rl-category/IQBench/data_verl_parquet_check/',
    '/mnt/cephfs/xubingye/wfs/datasets/rl-category/OCR-Reasoning/data_verl_parquet_check/',
  ]

# datasets_path = [
#     '/mnt/cephfs/xubingye/wfs/datasets/rl-category/EMMA/data_verl_parquet_check',
#     '/mnt/cephfs/xubingye/wfs/datasets/rl-category/zerobench/data_verl_parquet_check',
# ]

datasets_list = []
for dataset_path in datasets_path:
    assert os.path.exists(dataset_path), f'{dataset_path} not exists'
    for file_path in find_all_files(dataset_path, '.parquet'):
        datasets_list.append(file_path)

# breakpoint()
# if datasets_list:
#     combined_df = pd.concat(datasets_list, ignore_index=True)

dataframes = []
for parquet_file in datasets_list:
    df = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
    for item in df:
        cur_images = item['images']
        cur_prompt = item['prompt']
        assert len(cur_prompt) == 1
        if len(cur_images) > 1:
            breakpoint()
        # if not cur_prompt[0]['content'].count('<image>') == len(cur_images):
        #     breakpoint()

    # if "images" in dataframe.column_names:
        # def modify_images(example):
        #     for idx, image in enumerate(example["images"]):
        #         example["images"][idx]['path'] = ''
        #     return example
        # dataframe = dataframe.map(modify_images)
#     dataframes.append(dataframe)
# combine_dataset = datasets.concatenate_datasets(dataframes)
