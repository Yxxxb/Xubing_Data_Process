import pandas as pd
import os
import glob
import numpy as np
import re
from tqdm import tqdm
import datasets
from datakit.utils.files import dump_list_to_jsonl_file, find_all_files, read_jsonl_file
from datakit.utils.image import encode_bytes_to_base64_image

# data_path = "/mnt/cephfs/xubingye/wfs/datasets/rl-category/Clevr_CoGenT_TrainA_R1/data/train-00000-of-00015.parquet"
# tfile_path = '/mnt/cephfs/xubingye/wfs/datasets/rl-category/geometry3k_proc/train.parquet'
# vfile_path = '/mnt/cephfs/xubingye/wfs/datasets/rl-category/geometry3k_proc/test.parquet'
# new_path = "/mnt/cephfs/xubingye/wfs/datasets/rl-category/Clevr_CoGenT_TrainA_R1/data_verl_parquet/train-00000-of-00015.parquet"
# tdf = pd.read_parquet(tfile_path)
# vdf = pd.read_parquet(vfile_path)
# ndf = pd.read_parquet(new_path)
# data_df = pd.read_parquet(data_path)

# dataframet = datasets.load_dataset("parquet", data_files=tfile_path)["train"]
# dataframen = datasets.load_dataset("parquet", data_files=new_path)["train"]
# # print("###############", dataframe.iloc[0])

# breakpoint()

# sft_path = "/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_cot/ChartQA/data/grammar_correct_wo_gt_vlm_infer_Qwen72B_complexity_vllm_pe/00.jsonl"
# sft_path_cs = "/mnt/cephfs/xubingye/wfs/datasets/sft-category/Clevr_CoGenT_TrainA_R1_cold_start/data/grammar_correct/cold_start_1.jsonl"
# sft_data = read_jsonl_file(sft_path)
# sft_data_cs = read_jsonl_file(sft_path_cs)
# breakpoint()


# pip install pyarrow
parquets_path = "/mnt/cephfs/xubingye/wfs/datasets/rl-category/Clevr_CoGenT_TrainA_R1/data"
save_path = "/mnt/cephfs/xubingye/wfs/datasets/sft-category/Clevr_CoGenT_TrainA_R1_cold_start/data/grammar_correct"
parquet_files = []
for file in glob.glob(os.path.join(parquets_path, "*.parquet")):
    df = pd.read_parquet(file)
    parquet_files.append(df)

format_prompt = " You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."
image_placeholder = "<image>"

for i, df in enumerate(parquet_files):
    print(f"Processing file {i + 1}/{len(parquet_files)}: {df.shape}")
    save_lists = []
    for index, row in tqdm(df.iterrows()):
        save_item = {}
        answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', row['solution'])
        assert answer_match, "No answer found in the solution"
        answer = answer_match.group(1).strip()
        _c_id = f"Clevr_CoGenT_TrainA_R1_cold_start_{i}_{index}"
        save_item = {
            'id': _c_id,
            'base64_image': {_c_id: encode_bytes_to_base64_image(row['image']['bytes'])},
            'conversations': [
                {
                    'role': 'user',
                    'text': row['problem'] + format_prompt
                },
                {
                    'role': 'assistant',
                    'text': f"{row['thinking']}\\boxed{{{answer}}}"
                }
            ]
        }
        save_lists.append(save_item)
    dump_list_to_jsonl_file(os.path.join(save_path, f"cold_start_{i}.jsonl"), save_lists)
