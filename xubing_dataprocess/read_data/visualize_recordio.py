import os
import sys
import argparse
from transformers import AutoTokenizer
from datakit.utils.files import read_mmq_index

def read_mmq_recordio(path: str):
    from mmq_io.reader import Reader
    assert path.endswith(".recordio")
    return Reader(path)

file_path = "/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/OCR/Chinese-Card-OCR/recordio/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso/qwen_2_5_vl_72b_vllm_12M_reso/grammar_correct_vlm_infer_Qwen72B_vllm_12M_reso/0_0_data.recordio"
file_path = "/mnt/cephfs/bensenliu/wfs/vlmdatasets/MMPT/recordio/20250527e1/90_0_data.recordio"
reader = read_mmq_recordio(file_path)

breakpoint()
# qwen2vl detokenize
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/mnt/cephfs/bensenliu/wfs/weights/nlp/Qwen2.5-7B-Instruct")
def detokenize(text):
    return tokenizer.decode(text)

for data in reader:
    text = detokenize(data["text"].tolist())
    import ipdb
    ipdb.set_trace()
