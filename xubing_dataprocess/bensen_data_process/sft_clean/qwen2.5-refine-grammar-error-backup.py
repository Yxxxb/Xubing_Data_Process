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


points_sft = [
    # 'mm_ai2d',
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
    'sharegpt4v',
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


def extract_question_answer(response):
    conversations = []
    response = response.split('修复后的问题&答案对:')[1].strip()
    question_answers = response.split('user:')
    question_answers = [qa.strip()
                        for qa in question_answers if len(qa.strip()) > 0]
    question_answers = [question_answer.split(
        'assistant:') for question_answer in question_answers]
    question_answers = [[question_answer[0].strip(), question_answer[1].strip()]
                        for question_answer in question_answers]
    for question, answer in question_answers:
        conversations.extend([
            {
                'role': 'user',
                'text': question
            },
            {
                'role': 'assistant',
                'text': answer
            }
        ])
    return conversations


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
        # output folder
        grammar_error_refined_data_folder = os.path.join(
            data_root, sft, 'data/grammar_error_refined')
        os.makedirs(grammar_error_refined_data_folder, exist_ok=True)
        # output file
        grammar_error_refined_data_file = os.path.join(
            grammar_error_refined_data_folder, f'data_{rank}.jsonl')
        # if os.path.exists(grammar_error_refined_data_file):
        #     continue
        # input file
        grammar_error_data_cur_rank = os.path.join(
            grammar_error_data_folder, f'problem_data_{rank}.jsonl')
        data = read_jsonl_file(grammar_error_data_cur_rank)
        for item in tqdm(data):
            conversations = item['conversations']
            conversations_str = construct_conversations(conversations)
            prompt = PROMPT + '\n' + '###输入\n' + conversations_str + '\n\n###输出\n'
            response = model.generate(prompt, use_tqdm=False)
            yes_or_no = re.findall(yes_or_no_pattern, response)
            yes_or_no = yes_or_no[0].strip().lower()
            if 'no' in yes_or_no:
                results.append(item)
            else:
                conversations = extract_question_answer(response)
                item['conversations'] = conversations
                results.append(item)
        dump_list_to_jsonl_file(grammar_error_refined_data_file, results)

    print(f'Rank {rank} finished.')
    time.sleep(1000000)
