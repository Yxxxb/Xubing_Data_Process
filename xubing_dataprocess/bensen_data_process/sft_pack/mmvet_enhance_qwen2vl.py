import pandas as pd
from datakit import Qwen2VLLMWrapper
from datakit.utils.image import save_base64_image
from datakit.utils.distributed import get_distributed_env, dist_split_files
from datakit.utils.files import dump_list_to_jsonl_file
import os
from tqdm import tqdm


model_path = '/mnt/cephfs/bensenliu/wfs/weights/mm/opensource/qwen2-vl-72b-instruct'
model = Qwen2VLLMWrapper(model_path, max_tokens=2048,
                         tensor_parallel_size=8)


if __name__ == '__main__':  
    _, rank, world_size = get_distributed_env()
    data_file = '/mnt/cephfs/bensenliu/wfs/mm_datasets/eval/MMVet.tsv'
    output_file = f'/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/sft-enhanced/data/grammar_correct/sft-enhanced_{rank}.jsonl'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df = pd.read_csv(data_file, sep='\t')
    rows = [row.to_dict() for _, row in df.iterrows()]
    rows_cur_rank = dist_split_files(rows)
    results = []
    for i, row in enumerate(tqdm(rows_cur_rank)):
        sample_id = data_file + f'{rank}_{i}'
        image = row['image']
        question = row['question']
        image_tmp_path = f'/tmp/tmp_image_{rank}.jpg'
        save_base64_image(image, image_tmp_path)
        messages = [
            {
                'type': 'image',
                'content': image_tmp_path
            },
            {
                'type': 'text',
                'content': question
            }
        ]
        answer = model.generate(messages)
        template = {
            'id': sample_id,
            'base64_image': {
                sample_id: image
            },
            'conversations': [
                {'role': 'user', 'text': question},
                {'role': 'assistant', 'text': answer}
            ]
        }
        results.append(template)
    dump_list_to_jsonl_file(output_file, results)
    
    
