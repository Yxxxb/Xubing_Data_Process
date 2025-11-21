import json
import os
from argparse import ArgumentParser

import pandas as pd
import requests

parser = ArgumentParser()
parser.add_argument('--root',
                    type=str,
                    required=True,
                    help='root directory of the evaluation results')
parser.add_argument('--model-name', type=str, required=True)
parser.add_argument('--chat-id', type=str, required=True)
parser.add_argument('--exp-id', type=str, required=True)

dataset_evals = {
    'AI2D_TEST_acc.csv': ['Overall'],
    'HallusionBench_score.csv': ['aAcc', 'fAcc', 'qAcc'],
    'LLaVABench_score.csv': ['Relative Score (main)'],
    'MathVista_MINI_gpt-4o-mini_score.csv': ['acc'],
    'MMBench_DEV_EN_acc.csv': ['Overall'],
    'MMBench_TEST_CN_V11_acc.csv': ['Overall'],
    'MMBench_TEST_EN_V11_acc.csv': ['Overall'],
    'MME_score.csv': ['perception', 'reasoning'],
    'MMMU_DEV_VAL_acc.csv': ['Overall'],
    'MMStar_acc.csv': ['Overall'],
    'MMVet_gpt-4-turbo_score.csv': [[-1, 'acc']],
    'OCRBench_score.json': ['Final Score Norm'],
    'POPE_score.csv': ['Overall'],
    'RealWorldQA_acc.csv': ['Overall']
}

# dataset_evals = {
#     'OCRBench_score.json': ['Final Score Norm'],
# }


def compute_overall_score(results: dict) -> float:
    """Compute the overall score of the evaluation results.

    Args:
        results (dict): the evaluation results

    Returns:
        float: the overall score
    """
    over_scores = []
    datasets = [
        'AI2D_TEST_acc', 'HallusionBench_score',
        'MathVista_MINI_gpt-4o-mini_score', 'MMMU_DEV_VAL_acc', 'MMStar_acc',
        'MMVet_gpt-4-turbo_score', 'OCRBench_score'
    ]

    MMBench_score = round((results['MMBench_TEST_CN_V11_acc'] +
                           results['MMBench_TEST_EN_V11_acc']) / 2, 1)  # noqa
    over_scores.append(MMBench_score)
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
        eval_path = f'{root}/{model_name}_{key}'
        if not os.path.exists(eval_path):
            continue
        if 'OCR' in key:
            with open(eval_path) as f:
                data = json.load(f)
                key = key.split('.')[0]
                results[key] = data[val[0]]
        elif 'MMMU' in key:
            df = pd.read_csv(eval_path)
            res = df.iloc[0]['Overall'] if df.iloc[0][
                'split'] == 'validation' else df.iloc[1]['Overall']  # noqa
            res = round(res * 100, 1)
            key = key.split('.')[0]
            results[key] = res
        else:
            df = pd.read_csv(eval_path)
            cur_res = []
            for score_item in val:
                if isinstance(score_item, list):
                    cur_res.append(df.iloc[score_item[0]][score_item[1]])
                else:
                    cur_res.append(df[score_item].values[0])
            if 'MME' in key:
                cur_res = [cur_res[0] + cur_res[1]]
            res = sum(cur_res) / len(cur_res)
            if res < 1.0:
                res = res * 100
            res = round(res, 1)
            key = key.split('.')[0]
            results[key] = res
    average_score = compute_overall_score(results)
    results['Average'] = average_score
    results_str = exp_id + '\n\n'
    for key, val in results.items():
        results_str += f'{key}: {val}\n'
    return results_str.strip()


if __name__ == '__main__':
    args = parser.parse_args()
    results_str = extract_results(args.root, args.model_name, args.exp_id)
    web_hook = 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=bb2352cf-37b0-463d-9e1c-cb2a83a78a89'  # noqa
    headers = {'Content-Type': 'application/json'}
    data = {
        'chatid': args.chat_id,
        'msgtype': 'text',
        'text': {
            'content': results_str
        }
    }
    response = requests.post(web_hook, headers=headers, data=json.dumps(data))
    print(response.status_code)
    print(results_str)
