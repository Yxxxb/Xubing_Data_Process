from datakit.utils.files import read_jsonl_file, dump_list_to_jsonl_file
from datakit.utils.distributed import get_distributed_env
from datakit import vLLMWrapper
import os
import re
import time
from tqdm import tqdm

with open('points-1.2/prompts/grammar_refine.txt') as f:
    PROMPT = f.read()

model_path = '/mnt/cephfs/bensenliu/wfs/weights/nlp/Qwen2.5-72B-Instruct'
model = vLLMWrapper(model_path, 8, max_tokens=4096)

yes_or_no_pattern = r'是否存在语法问题:(.*)\n?'
question_pattern = r'user:(.*?)assistant:'


points_sft = [
    'mm_ai2d',
    'docvqa',
    'dvqa',
    'geoqa+',
    'allava_cap',
    'iconqa_choose_txt',
    'iconqa_fill_blank',
    'infovqa',
    'kvqa',
    'gpt4v',
    'llavar',
    'scienceqa',
    # 'sharegpt4v',
    'stvqa',
    'super_clever',
    'textvqa',
    'tqa',
    'vsr',
    'icdar_2015',
    'lima',
    'alpaca-gpt4',
    'openhermes2.5',
    'mini-gemini',
    'hme100k',
    'tabwp_cot',
    'geo3k',
    'clevr_math_5w',
    'poie',
    'lvis_instruct4v_cap',
    'MetaMathQA',
    'MathInstruct',
    'orca-math-word-problems-200k',
    'math',
    '500k-atlas-math',
    'gpt4o-complex-20240809-en',
    'MathV360K'
]


def construct_conversations(conversations):
    conversation_str = ''
    for conversation in conversations:
        role = conversation['role']
        text = conversation['text']
        conversation_str += f'{role}: {text}\n'
    return conversation_str.strip()


if __name__ == '__main__':
    _, rank, _ = get_distributed_env()
    success_datasets = []
    fail_datasets = []
    data_root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2'
    for sft in tqdm(points_sft):
        print(f'Processing {sft}...')
        results = []
        # input folder
        grammar_error_data_folder = os.path.join(
            data_root, sft, 'data/grammar_error')
        if not os.path.exists(grammar_error_data_folder):
            continue
        # output folder
        grammar_error_refined_data_folder = os.path.join(
            data_root, sft, 'data/grammar_error_refined')
        os.makedirs(grammar_error_refined_data_folder, exist_ok=True)
        # output file
        grammar_error_refined_data_file = os.path.join(
            grammar_error_refined_data_folder, f'data_{rank}.jsonl')
        if os.path.exists(grammar_error_refined_data_file):
            continue
        # input file
        grammar_error_data_cur_rank = os.path.join(
            grammar_error_data_folder, f'problem_data_{rank}.jsonl')
        if not os.path.exists(grammar_error_data_cur_rank):
            continue
        data = read_jsonl_file(grammar_error_data_cur_rank)
        for item in tqdm(data):
            refine_conversations = []
            conversations = item['conversations']
            for i in range(len(conversations)//2):
                conversation_turn = conversations[2*i:2*i+2]
                conversations_str = construct_conversations(conversation_turn)
                prompt = PROMPT + '\n' + '###输入\n' + conversations_str + '\n\n###输出\n'
                response = model.generate(prompt, use_tqdm=False)
                try:
                    yes_or_no = re.findall(yes_or_no_pattern, response)
                    yes_or_no = yes_or_no[0].strip().lower()
                    if 'no' in yes_or_no:
                        refine_conversations.extend(conversation_turn)
                    else:
                        question = re.search(
                            question_pattern, response, re.DOTALL).group(1).strip()
                        answer = response.split('assistant:')[-1].strip()
                        refine_conversations.append(
                            {'role': 'user', 'text': question})
                        refine_conversations.append(
                            {'role': 'assistant', 'text': answer})
                except Exception as e:
                    print(e)
                    continue
            if len(refine_conversations) > 0:
                item['conversations'] = refine_conversations
                results.append(item)
        dump_list_to_jsonl_file(grammar_error_refined_data_file, results)

    print(f'Rank {rank} finished.')
    time.sleep(1000000)
