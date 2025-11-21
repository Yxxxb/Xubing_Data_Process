from datakit.utils.files import read_jsonl_file, dump_list_to_jsonl_file
from datakit.utils.distributed import get_distributed_env, dist_split_files
from datakit import vLLMWrapper
import os
from tqdm import tqdm

model_path = '/mnt/cephfs/bensenliu/wfs/weights/nlp/Qwen1.5-32B-Chat'
model = vLLMWrapper(model_path, 8,
                    max_tokens=20,
                    temperature=0.7,
                    top_p=0.8,
                    repetition_penalty=1.05
                    )

points_sft = [
    'mm_ai2d',
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
    'scienceqa',
    # 'sharegpt4v',
    # 'stvqa',
    # 'super_clever',
    # 'textvqa',
    'tqa',
    # 'vsr',
    # 'icdar_2015',
    # 'hme100k',
    # 'tabwp_cot',
    # 'geo3k',
    # 'clevr_math_5w',
    # 'poie',
    # 'lvis_instruct4v_cap',
    # 'gpt4o-complex-20240809-en',
    # 'MathV360K'
]


if __name__ == '__main__':
    _, rank, _ = get_distributed_env()
    success_datasets = []
    fail_datasets = []
    data_root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2'
    points_sft_cur_rank = dist_split_files(points_sft)
    print(f'Rank {rank} processing {points_sft_cur_rank}')
    for sft in tqdm(points_sft_cur_rank):
        print(f'Processing {sft}...')
        input_file = f'{data_root}/{sft}/data/grammar_correct/{sft}.jsonl'
        text_image_folder = f'{data_root}/{sft}/data/grammar_correct_text_image'  # noqa
        os.makedirs(text_image_folder, exist_ok=True)
        text_image_results = []
        data = read_jsonl_file(input_file)
        for item in tqdm(data):
            conversations = item['conversations']
            new_conversations = []
            for i in range(len(conversations)//2):
                question_turn = conversations[2*i]
                answer_turn = conversations[2*i+1]
                answer = answer_turn['text']
                if len(answer) > 20:
                    continue
                question = question_turn['text']
                try:
                    prediction = model.generate(question, use_tqdm=False)
                    print(prediction)
                except:  # noqa
                    print(f'Error in generating prediction for {question}')
                    continue
                answer_turn['prediction'] = prediction
                new_conversations.append(question_turn)
                new_conversations.append(answer_turn)
            item['conversations'] = new_conversations
            text_image_results.append(item)
        if len(text_image_results) == 0:
            fail_datasets.append(sft)
        else:
            output_file = f'{text_image_folder}/{sft}.jsonl'
            dump_list_to_jsonl_file(output_file, text_image_results)
            success_datasets.append(sft)
    print(f'Success datasets: {success_datasets}')
    print(f'Fail datasets: {fail_datasets}')
