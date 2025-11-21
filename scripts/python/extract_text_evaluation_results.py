import argparse
import os

import pandas as pd

from datakit.utils.files import find_all_files
from datakit.utils.utils import send_messages_to_bot

parser = argparse.ArgumentParser()
parser.add_argument('--root',
                    type=str,
                    default='/mnt/cephfs/bensenliu/code/opencompass/outputs',
                    help='root directory of the evaluation results')
parser.add_argument('--chat-id', type=str, default='bensenliu')
parser.add_argument('--exp-id', type=str, required=True)
parser.add_argument('--message-type', type=str, default='markdown')
parser.add_argument(
    '--web-hook',
    type=str,
    default=  # noqa
    'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=bb2352cf-37b0-463d-9e1c-cb2a83a78a89'  # noqa
)


def extract_text_evaluation_results(evaluation_file: str, exp_id: str) -> str:
    datasets = [
        'GPQA_diamond', 'math', 'gsm8k', 'openai_humaneval', 'mmlu-weighted',
        'cmmlu', 'ceval'
    ]
    df = pd.read_csv(evaluation_file)
    evaluation_res_str = f'# POINTS-Omni {exp_id} Text Evaluation Results\n'
    evaluation_dict = {}
    for _, row in df.iterrows():
        evaluation_dict[row['dataset']] = row['qwen2.5-7b-instruct-hf']
    total_score, num_dataset = 0, 0
    for dataset in datasets:
        if dataset in evaluation_dict:
            evaluation_res_str += f'**{dataset}**: {evaluation_dict[dataset]}\n'  # noqa
            total_score += evaluation_dict[dataset]
            num_dataset += 1
    evaluation_res_str += f'**Overall**: {total_score/num_dataset:.2f}'  # noqa
    return evaluation_res_str.strip()


if __name__ == '__main__':
    args = parser.parse_args()
    evaluation_folder = os.path.join(args.root, args.exp_id[:6], args.exp_id,
                                     'summary')
    evaluation_file = find_all_files(evaluation_folder, 'csv')[0]
    evaluation_res_str = extract_text_evaluation_results(
        evaluation_file, args.exp_id)
    send_messages_to_bot(args.web_hook, args.message_type, args.chat_id,
                         evaluation_res_str)
