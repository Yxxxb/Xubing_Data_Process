import argparse
import json
import os

import numpy as np
import pandas as pd

SUPPORTED_MODEL_ALIAS = {
    'llama3': 'WeLLaVA-Local-LLaMA3-uhd',
    'yi-1.5': 'WeLLaVA-Local-Yi-1.5-uhd',
}
SUPPORTED_MODELS = list(SUPPORTED_MODEL_ALIAS.keys()) + list(
    SUPPORTED_MODEL_ALIAS.values())
LMU_DATA_PATH = '/mnt/cephfs/bensenliu/wfs/mm_datasets/eval'

parser = argparse.ArgumentParser('Mimikyu Evaluation Result Extractor')
parser.add_argument(
    '--model_version',
    type=str,
    required=True,
    help='checkpoint version, e.g. 20240810e1',
)
parser.add_argument(
    '--vlm_eval_run_py',
    type=str,
    default='/mnt/cephfs/ziyzhuang/code/VLMEvalKit/run.py',
    help='Path to VLMEvalKit run.py',  # noqa
)
parser.add_argument(
    '--ckpt_path',
    type=str,
    required=False,
    help='Model weights save path, saved as huggingface model',
)
parser.add_argument(
    '--model_type',
    choices=SUPPORTED_MODELS,
    required=False,
    help='Model type',
)
parser.add_argument(
    '--work_dir',
    type=str,
    default='/mnt/cephfs/ziyzhuang/exp_run/evaluation/llava/',
    help='Work directory, evaluation results will be saved here',  # noqa
)
args = parser.parse_args()


def process_ai2d(result_file: str):
    df = pd.read_csv(result_file)
    return df['Overall'].values[0] * 100


def process_pope(result_file: str):
    df = pd.read_csv(result_file)
    result = df[df['split'] == 'Overall']['Overall'].values[0]
    return result


def process_OCRBench(result_file: str):
    with open(result_file, 'r', encoding='utf-8') as f:
        result = json.load(f)
    return result['Final Score Norm']


def process_MMVet(result_file: str):
    df = pd.read_csv(result_file)
    result = df[df['Category'] == 'Overall']['acc'].values[0]
    return result


def process_HallusionBench(result_file: str):
    df = pd.read_csv(result_file)
    overall = df[df['split'] == 'Overall']
    scores = [
        overall[acc_type].values[0] for acc_type in ('aAcc', 'fAcc', 'qAcc')
    ]
    return np.mean(scores)


def process_MathVista(result_file: str):
    df = pd.read_csv(result_file)
    score = df[df['Task&Skill'] == 'Overall']['acc'].values[0]
    return score


def process_MMStar(result_file: str):
    df = pd.read_csv(result_file)
    return df['Overall'].values[0] * 100


def process_MMMU_Dev(result_file: str):
    df = pd.read_csv(result_file)
    score = df[df['split'] == 'validation']['Overall'].values[0]
    return score * 100


def process_MMBench_Dev(result_file: str):
    df = pd.read_csv(result_file)
    score = df['Overall'].values[0]
    return score * 100


DATASET_RESULT_PROCESS_FUNC_MAPPING = {
    'pope-overall': ('POPE_score.csv', process_pope),
    'OCRBench': ('OCRBench_score.json', process_OCRBench),
    'MMVet': ('MMVet_gpt-4-turbo_score_fine.csv', process_MMVet),
    'MMStar': ('MMStar_acc.csv', process_MMStar),
    'MMMU_Dev': ('MMMU_DEV_VAL_acc.csv', process_MMMU_Dev),
    'MMBench_TEST_EN_V11': (
        'MMBench_TEST_EN_V11.xlsx',
        lambda x: 'https://mmbench.opencompass.org.cn/mmbench-submission',
    ),
    'MMBench_TEST-CN_V11': (
        'MMBench_TEST_CN_V11.xlsx',
        lambda x: 'https://mmbench.opencompass.org.cn/mmbench-submission',
    ),
    'MMBench_Dev': ('MMBench_DEV_EN_acc.csv', process_MMBench_Dev),
    'MathVista': ('MathVista_MINI_gpt-4-turbo_score.csv', process_MathVista),
    'HallusionBench': ('HallusionBench_score.csv', process_HallusionBench),
    'ai2d': ('AI2D_TEST_acc.csv', process_ai2d),
}

model_name = args.model_type
if model_name in SUPPORTED_MODEL_ALIAS:
    model_name = SUPPORTED_MODEL_ALIAS[model_name]

input_path = os.path.join(args.work_dir, args.model_version, model_name)
check_file = os.path.join(
    input_path,
    f"{model_name}_{DATASET_RESULT_PROCESS_FUNC_MAPPING['MMBench_Dev'][0]}")

if not os.path.exists(check_file):
    if args.ckpt_path is None:
        args.ckpt_path = '[[YOUR_MODEL_CKPT_PATH_HF]]'
        print('\nPlease specify the model checkpoint path using --ckpt_path\n')
    # MMBench_TEST_EN_V11 and MMBench_TEST_CN_V11
    # are not included in the evaluation script
    work_dir = os.path.join(args.work_dir, args.model_version)
    cmd = (f'PYTHONPATH={args.ckpt_path} WELLAVA_MODEL_PATH={args.ckpt_path} '
           f'LMUData={LMU_DATA_PATH} '
           f'torchrun --nproc-per-node=8 {args.vlm_eval_run_py} '
           '--data MMMU_DEV_VAL MMBench_DEV_EN POPE HallusionBench '
           'OCRBench AI2D_TEST MMStar MMVet MathVista_MINI '
           'MME LLaVABench RealWorldQA '
           f'--model {model_name} '
           '--verbose '
           f'--work-dir {work_dir} ')
    print('YOU SHOULD RUN THE FOLLOWING COMMAND FIRST')
    print('==========================================')
    print(cmd)
    print('==========================================')
    exit(0)

results = dict()
for dataset, (file_suffix,
              func) in DATASET_RESULT_PROCESS_FUNC_MAPPING.items():
    result_file = os.path.join(input_path, f'{model_name}_{file_suffix}')
    if not os.path.exists(result_file):
        continue
    result = func(result_file)
    if isinstance(result, float):
        result = round(result, 1)
    results[dataset] = result
    print(f'{dataset:>20} - {result}')

output_path = os.path.join(input_path, 'eval_results.json')
with open(output_path, 'w+', encoding='utf-8') as f:
    results = json.dumps(results, indent=2)
    f.write(results)
