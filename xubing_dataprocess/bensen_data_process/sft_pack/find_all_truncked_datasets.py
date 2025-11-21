from tqdm import tqdm
import os
from datakit.utils.distributed import (dist_split_files,
                                       get_distributed_env,)
from datakit.utils.files import read_jsonl_file, dump_list_to_jsonl_file
from datakit.utils.mp import multi_process_with_append

points = [
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
    'stvqa',
    'super_clever',
    'textvqa',
    'tqa',
    'vsr',
    'icdar_2015',
    'hme100k',
    'tabwp_cot',
    'geo3k',
    'clevr_math_5w',
    'poie',
    'lvis_instruct4v_cap',
    'gpt4o-complex-20240809-en',
    'MathV360K',
    'mapqa',
    'sharegpt4v_choice_cn',
    'sharegpt4v_choice',
    'sharegpt4v_llava158k',
    'sharegpt4v_qa',
    'sharegpt4v_ref',
    'sharegpt4v_sharegpt',
    'sharegpt4v_textcaps',
    'sharegpt4v_qa_cn',
    'memes-500',
    'Chinese-OCR',
    'LaTeX_OCR',
    'LaTeX_OCR_column_rename_full',
    'sft-enhanced',
]



if __name__ == '__main__':
    filenames_cur_rank = dist_split_files(points)
    world_size, rank, _ = get_distributed_env()
    output_datasets = []
    for filename in tqdm(filenames_cur_rank):
        output_file = f'/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/{filename}/data/grammar_correct_trunck/{filename}.jsonl'
        if os.path.exists(output_file):
            output_datasets.append(filename)
    print(output_datasets)
      

        
