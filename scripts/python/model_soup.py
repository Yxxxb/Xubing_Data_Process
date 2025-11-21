import argparse
import os
from collections import OrderedDict

from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--model-paths', type=str, nargs='+', required=True)
parser.add_argument('--output-path', type=str, required=True)

if __name__ == '__main__':
    args = parser.parse_args()
    model_paths = args.model_paths
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    state_dicts = []
    for model_path in model_paths:
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     trust_remote_code=True)
        state_dict = model.state_dict()
        state_dicts.append(state_dict)
    tokenizer = AutoTokenizer.from_pretrained(model_paths[0],
                                              trust_remote_code=True,
                                              use_fast=False)
    weight_keys = list(state_dicts[0].keys())
    final_state_dict = OrderedDict()
    for key in weight_keys:
        key_sum = 0
        for state_dict in state_dicts:
            key_sum += state_dict[key]
        final_state_dict[key] = key_sum / len(state_dicts)
    model.load_state_dict(final_state_dict, strict=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
