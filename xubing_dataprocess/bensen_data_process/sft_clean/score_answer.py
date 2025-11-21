from datakit.utils.files import read_jsonl_file, dump_list_to_jsonl_file
from datakit.utils.distributed import get_distributed_env, dist_split_files
from datakit import vLLMWrapper
from tqdm import tqdm
import os

model_path = '/mnt/cephfs/bensenliu/wfs/weights/nlp/Qwen1.5-32B-Chat'
model = vLLMWrapper(model_path, 8,
                    max_tokens=20,
                    temperature=0.7,
                    top_p=0.8,
                    repetition_penalty=1.05
                    )
with open('points-1.2/prompts/score_answer.txt') as f:
    PROMPT = f.read()

points_sft = [
    'mm_ai2d',
    'scienceqa',
    'tqa'
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
        input_file = f'{data_root}/{sft}/data/grammar_correct_text_image/{sft}.jsonl'  # noqa
        image_text_output_file = input_file
        text_only_output_file = input_file.replace('grammar_correct_text_image', 'grammar_correct_text_only')  # noqa
        os.makedirs(os.path.dirname(text_only_output_file), exist_ok=True)
        data = read_jsonl_file(input_file)
        image_text_results = []
        text_only_results = []
        for item in tqdm(data):
            conversations = item['conversations']
            text_only_conversations = []
            image_text_conversations = []
            for i in range(len(conversations)//2):
                question_turn = conversations[2*i]
                answer_turn = conversations[2*i+1]
                question = question_turn['text']
                target_answer = answer_turn['text']
                predicted_answer = answer_turn['prediction']
                prompt = PROMPT + '\n' + question + \
                    f'### Standard answer\n{target_answer}\n' + \
                    f'### Model answer\n{predicted_answer}\n' + \
                    '### Score:\n'
                try:
                    response = model.generate(prompt, use_tqdm=False).lower()
                except:  # noqa
                    print(f'Error generating response for {question}')
                    continue
                if 'yes' in response:
                    text_only_conversations.extend([
                        question_turn, answer_turn
                    ])
                else:
                    image_text_conversations.extend([
                        question_turn, answer_turn
                    ])
            if len(text_only_conversations) > 0:
                text_only_item = item.copy()
                text_only_item['conversations'] = text_only_conversations
                text_only_item['base64_image'] = {}
                text_only_results.append(text_only_item)
            if len(image_text_conversations) > 0:
                image_text_item = item.copy()
                image_text_item['conversations'] = image_text_conversations
                image_text_results.append(image_text_item)
        if len(text_only_results) == 0 and len(image_text_results) == 0:
            fail_datasets.append(sft)
        else:
            success_datasets.append(sft)
        dump_list_to_jsonl_file(text_only_output_file, text_only_results)
        dump_list_to_jsonl_file(image_text_output_file, image_text_results)
    print(f'Rank {rank} success datasets: {success_datasets}')
    print(f'Rank {rank} fail datasets: {fail_datasets}')
