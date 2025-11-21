import argparse
import base64
import json
import os
from typing import List, Optional, Tuple


def find_all_files(
        root_dir,
        extension):
    """Find all files with the given extension in the given directory and its
    subdirectories.

    Args:
        root_dir (str): The root directory to search in.
        extension Optional(str|Tuple[str]):
            The extension of the files to search for. Defaults to 'tar'.

    Returns:
        list: A list of all files with the given extension in the
            given directory. The full path will be included.
    """
    files = []
    if isinstance(extension, str):
        extension = (extension, )
    else:
        extension = tuple(extension)
    for root, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(extension):
                files.append(os.path.join(root, filename))
    return files

def read_jsonl_file(file_path: str):
    """Read a JSONL file and return a list of objects.

    Args:
        file_path (str): The path of the JSONL file to be read.

    Returns:
        List[any]: A list of objects read from the JSONL file.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        # objects = [json.loads(line.strip()) for line in lines]
    # return objects
    return lines


# lens = 0
# with open("/mnt/cephfs/xubingye/MMDataKit/workspace/data_to_filter.txt", 'r') as f:
#     datasets = f.readlines()
#     datasets = [dataset.strip() for dataset in datasets]
#     for dataset in datasets:
#         all_jsonls = find_all_files(dataset, 'jsonl')
#         for jsonl in all_jsonls:
#             data = read_jsonl_file(jsonl)
#             lens += len(data)
#         print(dataset, lens)

# print(lens)


import jsonlines

# 校验是否完成inference
with open("/mnt/cephfs/xubingye/vlm/MMDataKit/workspace/data_to_filter.txt", 'r') as f:
    datasets = f.readlines()
    datasets = [dataset.strip() for dataset in datasets]
    for idx, dataset in enumerate(datasets):
        # if idx <= 17:
        #     continue
        all_jsonls = find_all_files(dataset, 'jsonl')
        assert len(all_jsonls) == 1
        objs = []
        with jsonlines.open(all_jsonls[0], 'r') as reader:
            for obj in reader:
                objs.append(obj)

        all_jsonls = find_all_files(dataset + "x", 'jsonl')
        objs_vllm = []
        none_num = 0
        for jsonl in all_jsonls:
            cur_objs_vllm = []
            with jsonlines.open(jsonl, 'r') as reader:
                for obj in reader:
                    cur_objs_vllm.append(obj)
            assert all([len(obj['conversations']) == 2 for obj in cur_objs_vllm])
            assert all(['qwen2.5vl-72b' in obj['conversations'][1] for obj in cur_objs_vllm])
            # assert all([obj['conversations'][1]['qwen2.5vl-72b'] is not None for obj in cur_objs_vllm])
            none_num += sum([obj['conversations'][1]['qwen2.5vl-72b'] is None for obj in cur_objs_vllm])
            objs_vllm += cur_objs_vllm

        assert len(objs) == len(objs_vllm)
        # breakpoint()
        print(idx, dataset, len(objs), len(objs_vllm), none_num)
        

# objs = []
# with jsonlines.open("/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/OCR/Chinese-OCR/data/grammar_correct/Chinese-OCR.jsonl", 'r') as reader:
#     for obj in reader:
#         objs.append(obj)

# objs_vllm = []
# all_jsonls = find_all_files("/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/OCR/Chinese-OCR/data/grammar_correct_vlm_infer_Qwen72B_vllm", 'jsonl')
# for jsonl in all_jsonls:
#     cur_objs_vllm = []
#     with jsonlines.open(jsonl, 'r') as reader:
#         for obj in reader:
#             cur_objs_vllm.append(obj)
#     assert all([len(obj['conversations']) == 2 for obj in cur_objs_vllm])
#     assert all(['qwen2.5vl-7b' in obj['conversations'][1] for obj in cur_objs_vllm])
#     objs_vllm += cur_objs_vllm

# breakpoint()

# with jsonlines.open("/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/OCR/Chinese-OCR/data/grammar_correct_vlm_infer_Qwen72B_vllm/00.jsonl", 'r') as reader:
#     for obj in reader:
#         assert 'qwen2.5vl-7b' in obj['conversations'][1]
#         assert len(obj['conversations']) == 2
#     # breakpoint()


"""
1. 检查json的长度、输出格式
2. 可视化
3. 修改一下7B：
4. 训练一下
5. 3B的bug看一下
"""
