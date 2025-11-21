from huggingface_hub import HfApi

api = HfApi()

api.upload_large_folder(
    repo_id='WePOINTS/POINTS-Qwen-2-5-7B-Chat',
    repo_type='model',
    folder_path='/mnt/cephfs/pr-nlp-large_pretrain/bensenliu/exp_runs/weights/mm/mmq-llava-20241013e1-20241014e1-20241014e2-20241014e3-average-sft-hf'
)
