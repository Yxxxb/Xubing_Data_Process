import os
import sys
import argparse
# from transformers import AutoTokenizer
# from datakit.utils.files import read_mmq_index, read_mmq_recordio
from datakit.utils.files import dump_list_to_jsonl_file, find_all_files, read_jsonl_file, mem_efficient_read_jsonl_file, read_parquet_file

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

with open("/mnt/cephfs/xubingye/wfs/weights/vlm/mmq-pointsv15-20250707e1xubing-sft-hf/tokenizer_config.json", 'r', encoding='utf-8') as file:
    data = json.load(file)
    print(data)  # 打印读取的 JSON 数据

file = read_jsonl_file("/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/SCIENCE/mm_ai2d/data/grammar_correct_split_3k_caption_prompt/455.jsonl")
file = read_jsonl_file("/mnt/cephfs/bensenliu/wfs/vlmdatasets/pt/PDF/tiku/data/screenshots_gzsw/part-00562-49e94dbc-de58-47ee-8b14-0d3b30067b62-c000.json")
parq = read_parquet_file("/mnt/cephfs/xubingye/wfs/datasets/rl-category/MMK12/data_verl_parquet/train-00000-of-00004.parquet")
# test = read_jsonl_file("/mnt/cephfs/xubingye/wfs/weights/vlm/mmq-pointsv15-20250707e1xubing-sft-hf/tokenizer_config.json")
breakpoint()

# file1 = mem_efficient_read_jsonl_file("/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/GROUNDING/sharegpt4v_ref/data/grammar_correct/sharegpt4v_ref.jsonl")
# file1 = mem_efficient_read_jsonl_file("/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/CHART/ChartQA/data/grammar_correct_single_word/ChartQA.jsonl")
# file1 = mem_efficient_read_jsonl_file("/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/DIAGRAM/mapqa/data/grammar_correct/mapqa.jsonl")
# file1 = mem_efficient_read_jsonl_file("/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_cot/scienceqa/data/grammar_correct_wo_gt/scienceqa.jsonl")
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

caption_prompt = "You need to generate a **detailed** description of the image based on the image and the question.\n "\
                "The description needs to be as comprehensive as possible, focusing on the **overall content of the image** and the **details of all the objects**. "\
                "Your description will be used to answer the question provided, so it is also important that your description contains as much detail as possible about what is involved in answering the question.\n\n "\
                "For example, for images rich in text and tables, you need to extract the entire content. "\
                "For images containing multiple objects, you need to give not only a detailed description of each object, but also a description of the image as a whole and the positional relationships between the objects. "\
                "For mathematical graphs, you need to give a detailed description of the mathematical graph in the context of the problem as much as possible.\n\n "\
                


_num_sum = 0
for dataset in datasets:
    dataset += '_split_3k'
    output_folder = dataset + '_caption_prompt'
    os.makedirs(output_folder, exist_ok=True)

    _num_data = 0
    save_3k_data = []
    all_jsonls = find_all_files(dataset, "jsonl")
    for jsonl in all_jsonls:
        jsonl_data = read_jsonl_file(jsonl)
        for i in jsonl_data:
            assert len(i['conversations']) == 2
            i['conversations'][0]['text'] = "You need to generate a **detailed** description of the image based on the image and the question.\n "\
                "The description needs to be as comprehensive as possible, focusing on the **overall content of the image** and the **details of all the objects**. "\
                "Your description will be used to answer the question provided, so it is also important that your description contains as much detail as possible about what is involved in answering the question.\n\n "\
                "For example, for images rich in text and tables, you need to extract the entire content. "\
                "For images containing multiple objects, you need to give not only a detailed description of each object, but also a description of the image as a whole and the positional relationships between the objects. "\
                "For mathematical graphs, you need to give a detailed description of the mathematical graph in the context of the problem as much as possible.\n\n "\
                f"The question is '{i['conversations'][0]['text']}'.\n\n "\
                "Please provide description for the image below as requested: "

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
