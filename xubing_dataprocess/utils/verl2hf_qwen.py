#!/usr/bin/env python
# encoding: utf-8
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
from glob import glob
from collections import defaultdict
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model-id', type=str, help='Id of model.')
parser.add_argument('--model-step', type=str, help='Step of model.')

def main():
    args = parser.parse_args()
    path = f"/mnt/cephfs/xubingye/vlm/rl/v0528_rl/checkpoints/wepoints_rl/wepoints_rl_xubing_{args.model_id}/global_step_{args.model_step}"
    if path == "do_not_convert":
        return

    parts = [i.strip() for i in path.split("/") if i.strip()]
    global_step = parts[-1]
    if not global_step.startswith("global_step_"):
        print("verl ckpt should endswith global_step_*")
        return
    
    root = path
    fsdp_checkpoint_path = f"{root}/actor"
    huggingface_model_path = f"{root}/actor/huggingface"
    output_path = f"/mnt/cephfs/xubingye/wfs/weights/rl/mmq-pointsv15-{args.model_id}-{args.model_step}-rl/"
    state_dict = defaultdict(list)

    ### check if the model has been converted
    try:
        config = AutoConfig.from_pretrained(output_path, trust_remote_code=True)
        print("the model has been converted, pass")
        return
    except:
        pass
    
    print("begin convert")
    try:
        config = AutoConfig.from_pretrained(huggingface_model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    except Exception as e:
        print(e)
        print(f"no huggingface_model_path in {huggingface_model_path}, use default")
        huggingface_model_path = '/mnt/cephfs/bensenliu/wfs/weights/mm/opensource/Qwen2.5-VL-7B-Instruct'
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    all_file_path = glob(f"{fsdp_checkpoint_path}/model_world_size_*_rank_*.pt")
    world_size = len(all_file_path)
    for rank in range(world_size):
        filepath = f"{fsdp_checkpoint_path}/model_world_size_{world_size}_rank_{rank}.pt"
        print('loading', filepath)
        this_state_dict = torch.load(filepath, weights_only=False)
        for key, value in this_state_dict.items():
            state_dict[key].append(value.to_local())

    for key in state_dict:
        state_dict[key] = torch.cat(state_dict[key], dim=0)


    model.load_state_dict(state_dict)

    model.save_pretrained(output_path, max_shard_size="10GB")

    tokenizer = AutoTokenizer.from_pretrained(huggingface_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    print(f"save to {output_path}")


if __name__ == "__main__":
    main()
