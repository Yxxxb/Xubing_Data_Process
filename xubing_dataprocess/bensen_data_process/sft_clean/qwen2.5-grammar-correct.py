from vllm import LLM, SamplingParams
import os
import time
from tqdm import tqdm
from transformers import AutoTokenizer
from datakit.utils.files import read_jsonl_file, find_all_files, dump_list_to_jsonl_file
from datakit.utils.distributed import dist_split_files, get_distributed_env


with open('points-1.2/prompts/bad_question_cn.txt') as f:
    PROMPT = f.read().strip()


model_path = '/mnt/cephfs/bensenliu/wfs/weights/nlp/Qwen2.5-72B-Instruct'
llm = LLM(model_path, trust_remote_code=True,
          tensor_parallel_size=8)
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1024)
tokenizer = AutoTokenizer.from_pretrained(model_path)


points_sft = [
    # 'mm_ai2d',
    # 'docvqa',
    # 'dvqa',
    # 'geoqa+',
    # 'allava_cap',
    # 'iconqa_choose_txt',
    # 'iconqa_fill_blank',
    # 'infovqa',
    # 'kvqa',
    # 'gpt4v',
    # 'llavar',
    # 'scienceqa',
    # 'sharegpt4v',
    # 'stvqa',
    # 'super_clever',
    # 'textvqa',
    # 'tqa',
    # 'vsr',
    # 'icdar_2015',
    # 'lima',
    # 'alpaca-gpt4',
    # 'openhermes2.5',
    # 'mini-gemini',
    # 'hme100k',
    # 'tabwp_cot',
    # 'geo3k',
    # 'clevr_math_5w',
    # 'poie',
    # 'lvis_instruct4v_cap',
    # 'MetaMathQA',
    # 'MathInstruct',
    # 'orca-math-word-problems-200k',
    # 'math',
    # '500k-atlas-math',
    # 'gpt4o-complex-20240809-en',
    # 'MathV360K',
    'mapqa'
]


def construct_conversations(conversations):
    conversation_str = ''
    for conversation in conversations:
        role = conversation['role']
        text = conversation['text']
        conversation_str += f'{role}: {text}\n'
    return conversation_str.strip()


if __name__ == '__main__':
    root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft'
    output_root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2'
    world_size, rank, local_rank = get_distributed_env()
    for dataset in tqdm(points_sft):
        print(f'processing {dataset}...')
        cur_folder = f'{root}/{dataset}/data'
        output_folder = f'{output_root}/{dataset}/data'
        output_file = f'{output_folder}/data_{rank}.jsonl'
        problem_output_file = f'{output_folder}/problem_data_{rank}.jsonl'
        if os.path.exists(output_file):
            print(f'{output_file} exists, skipped')
            continue
        os.makedirs(output_folder, exist_ok=True)
        try:
            dataset_file = find_all_files(cur_folder, 'jsonl')[0]
        except Exception as e: # noqa
            print(f'no data found for {dataset}, skipped')
            continue
        data = read_jsonl_file(dataset_file)
        data_cur_rank = dist_split_files(data)
        results = []
        problem_results = []
        for item in tqdm(data_cur_rank):
            conversations = item['conversations']
            conversations_str = construct_conversations(conversations)
            prompt = f'{PROMPT}\n{conversations_str}'
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            resp = llm.generate(text, sampling_params, use_tqdm=False)
            resp = resp[0].outputs[0].text.lower()
            if 'yes' in resp:
                problem_results.append(item)
            else:
                results.append(item)
        dump_list_to_jsonl_file(output_file, results)
        dump_list_to_jsonl_file(problem_output_file, problem_results)
    print('done')
    time.sleep(1000000)
