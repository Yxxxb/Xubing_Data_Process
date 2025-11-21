import json
import os
from argparse import ArgumentParser

import pandas as pd
import requests

parser = ArgumentParser()
parser.add_argument(
    '--root',
    type=str,
    default='/mnt/cephfs/bensenliu/mm_eval/eval/eval_res/202503/20250327e5-64',
    help='root directory of the evaluation results')
parser.add_argument('--model-name', type=str, default='POINTSV15-API')
parser.add_argument('--chat-id', type=str, default='')
parser.add_argument('--exp-id', type=str, default='POINTS15')
parser.add_argument(
    '--web-hook',
    type=str,
    default="""https://qyapi.weixin.qq.com/cgi-bin/webhook/send?
               key=bb2352cf-37b0-463d-9e1c-cb2a83a78a89""")

dataset_evals = {
    'LongVideoBench_64frame_nopack_nosubs_rating.json': 'overall',
    'MLVU_64frame_nopack_acc.csv': 'M-Avg',
    'MMBench-Video_64frame_nopack_gpt-4-turbo_rating.json': 'coarse_all',
    'MVBench_64frame_nopack_rating.json': 'overall',
    'TempCompass_64frame_nopack_acc.csv': 'overall',
    'Video-MME_64frame_nopack_nosubs_rating.json': 'overall',
}


def compute_overall_score(results: dict) -> float:
    """Compute the overall score of the evaluation results.

    Args:
        results (dict): the evaluation results

    Returns:
        float: the overall score
    """
    over_scores = []
    datasets = [
        'LongVideoBench',
        'MLVU',
        'MMBench-Video',
        'MVBench',
        'TempCompass',
        'Video-MME',
    ]
    for dataset in datasets:
        over_scores.append(results[dataset])
    return round(sum(over_scores) / len(over_scores), 1)


def extract_results(root: str, model_name: str, exp_id: str) -> str:
    """Extract the evaluation results.

    Args:
        root (str): root directory of the evaluation results
        model_name (str): name of the model
        exp_id (str): experiment id

    Returns:
        str: the extracted evaluation results
    """
    results = {}
    for key, val in dataset_evals.items():
        eval_path = f'{root}/{model_name}/{model_name}_{key}'
        if not os.path.exists(eval_path):
            continue
        if 'LongVideoBench' in key:
            with open(eval_path) as f:
                data = json.load(f)
                key = 'LongVideoBench'
                results[key] = float(data[val]['overall']) * 100
        elif 'MLVU' in key:
            df = pd.read_csv(eval_path)
            res = float(df.iloc[9]['acc'])
            key = 'MLVU'
            results[key] = res
        elif 'MMBench-Video' in key:
            with open(eval_path) as f:
                data = json.load(f)
                key = 'MMBench-Video'
                results[key] = float(data[val]['Overall'])
        elif 'MVBench' in key:
            with open(eval_path) as f:
                data = json.load(f)
                key = 'MVBench'
                results[key] = float(data[val][-1][:-1])
        elif 'TempCompass' in key:
            df = pd.read_csv(eval_path, index_col=0)
            res = float(df.loc[val]['acc'])
            key = 'TempCompass'
            results[key] = res
        elif 'Video-MME' in key:
            with open(eval_path) as f:
                data = json.load(f)
                key = 'Video-MME'
                results[key] = float(data[val]['overall']) * 100
    average_score = compute_overall_score(results)
    results['Average'] = average_score
    results_str = exp_id + '\n\n'
    for key, val in results.items():
        results_str += f'{key}: {val}\n'
    return results_str.strip()


if __name__ == '__main__':
    args = parser.parse_args()
    results_str = extract_results(args.root, args.model_name, args.exp_id)
    web_hook = args.web_hook  # noqa
    headers = {'Content-Type': 'application/json'}
    data = {
        'chatid': args.chat_id,
        'msgtype': 'text',
        'text': {
            'content': results_str
        }
    }
    print(results_str)
    with open(os.path.join(args.root, 'results.txt'), 'w') as f:
        f.write(results_str)
    response = requests.post(web_hook, headers=headers, data=json.dumps(data))
    print(response.status_code)
    print(results_str)
