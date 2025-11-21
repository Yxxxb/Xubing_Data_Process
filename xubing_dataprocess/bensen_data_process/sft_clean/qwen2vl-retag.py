from datakit import Qwen2VLLMWrapper
from datakit.utils.files import read_jsonl_file, dump_list_to_jsonl_file
from datakit.utils.distributed import dist_split_files, get_distributed_env
from datakit.utils.image import save_base64_image
from tqdm import tqdm
import os
import time

datasets = [
    ['gpt4v', '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/gpt4v/data/grammar_correct/gpt4v.jsonl'],
    ['sharegpt4v', '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/sharegpt4v/data/grammar_correct_caption/sharegpt4v_caption.jsonl'],
    ['allava_cap', '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/allava_cap/data/grammar_correct/allava_cap.jsonl'],
]

model_path = '/mnt/cephfs/bensenliu/wfs/weights/mm/opensource/qwen2-vl-72b-instruct'
model = Qwen2VLLMWrapper(model_path, max_tokens=1024,
                         tensor_parallel_size=8)


def recap_single_item(item, rank):
    image_key = list(item['base64_image'].keys())[0]
    base64_image = item['base64_image'][image_key]
    tmp_path = f'/tmp/{rank}.jpg'
    save_base64_image(base64_image, tmp_path)
    conversations = item['conversations']
    for i in range(len(conversations)//2):
        question = conversations[2*i]['text']
        messages = [
            {
                'type': 'image',
                'content': tmp_path
            },
            {
                'type': 'text',
                'content': question
            }
        ]
        answer = model.generate(messages)
        conversations[2*i+1]['text'] = answer
    os.remove(tmp_path)
    item['conversations'] = conversations
    return item


if __name__ == '__main__':
    root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2'
    _, rank, _ = get_distributed_env()
    for dataset in datasets:
        dataset_name = dataset[0]
        dataset_path = dataset[1]
        output_file = f'{root}/{dataset_name}/data/grammar_correct_qwen2vl_72b/{dataset_name}_{rank}.jsonl'
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        data = read_jsonl_file(dataset_path)
        data_cur_rank = dist_split_files(data)
        results = []
        for item in tqdm(data_cur_rank):
            try:
                item = recap_single_item(item, rank)
            except Exception as e:
                continue
            results.append(item)
        dump_list_to_jsonl_file(output_file, results)
        print(f'Finish {dataset_name} {rank}')
    print('Finish!!!!!!')
    time.sleep(1000000)


