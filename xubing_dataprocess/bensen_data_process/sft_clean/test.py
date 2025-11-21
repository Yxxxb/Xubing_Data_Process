import os
from tqdm import tqdm

sharegpt4v_qa_path = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/sharegpt4v_qa/data/grammar_correct'
sharegpt4v_choice_path = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/sharegpt4v_choice/data/grammar_correct'
sharegpt4v_ref_path = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/sharegpt4v_ref/data/grammar_correct'
sharegpt4v_textcaps_path = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/sharegpt4v_textcaps/data/grammar_correct'
sharegpt4v_llava158k_path = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/sharegpt4v_llava158k/data/grammar_correct'
sharegpt4v_sharegpt_path = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/sharegpt4v_sharegpt/data/grammar_correct'

paths = [sharegpt4v_qa_path, sharegpt4v_choice_path, sharegpt4v_ref_path, 
         sharegpt4v_textcaps_path, sharegpt4v_llava158k_path, sharegpt4v_sharegpt_path]

if __name__ == '__main__':
    for path in tqdm(paths):
        filename = path.split('/')[-3] + '.jsonl'
        filepath = os.path.join(os.path.dirname(path), filename)
        os.system(f'mv {path} {filepath}')
        os.makedirs(path, exist_ok=True)
        os.system(f'mv {filepath} {path}')
