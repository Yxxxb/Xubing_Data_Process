import os
import sys
import argparse
# from transformers import AutoTokenizer
# from datakit.utils.files import read_mmq_index, read_mmq_recordio
from datakit.utils.files import dump_list_to_jsonl_file, find_all_files, read_jsonl_file, mem_efficient_read_jsonl_file

import json

"""
/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/CHART/ChartQA/data/grammar_correct 32719
/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/DIAGRAM/infovqa/data/grammar_correct 24428
/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/DIAGRAM/mapqa/data/grammar_correct 24720 先不做
/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/GROUNDING/sharegpt4v_ref/data/grammar_correct 79622 
/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/MATH/clevr_math_5w/data/grammar_correct 48224
/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/MATH/geo3k/data/grammar_correct 2051
/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/MATH/geoqa+/data/grammar_correct 68145
/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/MATH/iconqa_choose_txt/data/grammar_correct 18869
/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/MATH/iconqa_fill_blank/data/grammar_correct 10913
/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/MATH/super_clever/data/grammar_correct 254838
/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/SCIENCE/mm_ai2d/data/grammar_correct 455
/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/SCIENCE/scienceqa/data/grammar_correct 6149
/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/SCIENCE/tqa/data/grammar_correct 8978
/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/MATH/MathV360K/data/grammar_correct 325535 先不做
580111
"""

# file1 = mem_efficient_read_jsonl_file("/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/GROUNDING/sharegpt4v_ref/data/grammar_correct/sharegpt4v_ref.jsonl")
# file1 = mem_efficient_read_jsonl_file("/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/CHART/ChartQA/data/grammar_correct_single_word/ChartQA.jsonl")
# file1 = mem_efficient_read_jsonl_file("/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/DIAGRAM/mapqa/data/grammar_correct/mapqa.jsonl")
# count = 0
# for i in file1:
#     print(i)
#     count += 1
#     breakpoint()
# print(count)
# breakpoint()

output_txt_path = "/mnt/cephfs/xubingye/vlm/MMDataKit/workspace/complex_data_to_filter.txt"
with open(output_txt_path, "r") as f:
    datasets = f.readlines()
datasets = [dataset.strip() for dataset in datasets]

_num_sum = 0
for dataset in datasets:
    output_folder = dataset + '_split_3k'
    os.makedirs(output_folder, exist_ok=True)

    _num_data = 0
    save_3k_data = []
    all_jsonls = find_all_files(dataset, "jsonl")
    for jsonl in all_jsonls:
        jsonl_data = mem_efficient_read_jsonl_file(jsonl)
        for i in jsonl_data:
            if "sharegpt4v_ref" in dataset or "mapqa" in dataset:
                assert len(i['conversations']) % 2 == 0
                for conv_idx in range(0, len(i['conversations']), 2):
                    _i = {
                        'id': i['id'],
                        'base64_image': i['base64_image'],
                        'conversations': i['conversations'][conv_idx:conv_idx+2]
                    }
                    save_3k_data.append(_i)
                    _num_data += 1
                    if _num_data % 3000 == 0:
                        dump_list_to_jsonl_file(os.path.join(output_folder, f"{_num_data}.jsonl"), save_3k_data)
                        save_3k_data = []
            else:
                save_3k_data.append(i)
                _num_data += 1
                if _num_data % 3000 == 0:
                    dump_list_to_jsonl_file(os.path.join(output_folder, f"{_num_data}.jsonl"), save_3k_data)
                    save_3k_data = []
    if save_3k_data != []:
        dump_list_to_jsonl_file(os.path.join(output_folder, f"{_num_data}.jsonl"), save_3k_data)

    _num_sum += _num_data
    print(f"{dataset} {_num_data}")
print(_num_sum)
