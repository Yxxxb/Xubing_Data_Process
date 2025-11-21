from datakit.utils.files import dump_list_to_jsonl_file
from datakit.utils.image import encode_pil_to_base64_image
from tqdm import tqdm
from datasets import load_dataset


if __name__ == '__main__':
    ori_file = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/ChineseOCRBench/'
    dataset = load_dataset(ori_file)
    output_file = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/ChineseOCRBench/data/ChineseOCRBench.jsonl'
    results = []
    for meta in tqdm(dataset['test']):
        image_name = ori_file + str(meta['id'])
        base64_image = encode_pil_to_base64_image(meta['image'])
        question = meta['question']
        answer = meta['answers']
        template = {
                'id': image_name,
                'base64_image': {
                    image_name: base64_image
                },
                'conversations': [
                    {
                        'role': 'user',
                        'text': question
                    },
                    {
                        'role': 'assistant',
                        'text': answer
                    }
                ]
            }
        results.append(template)
    dump_list_to_jsonl_file(output_file, results)
    
        
