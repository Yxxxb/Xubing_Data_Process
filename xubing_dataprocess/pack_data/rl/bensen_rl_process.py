import pandas as pd
from tqdm import tqdm
from PIL import Image
import io
from datakit.utils.image import decode_base64_image_to_pil




if __name__ == '__main__':
    train_test_split = 0.1
    ori_path = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/rl/general/ccbench/CCBench.tsv'
    output_root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/rl/general/ccbench'
    df = pd.read_csv(ori_path, sep='\t')
    train_results = []
    test_results = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        if i < int(len(df) * (1 - train_test_split)):
            split = 'train'
        else:
            split = 'test'
        prompt = row['question']
        for choice in ['A', 'B', 'C', 'D']:
            if choice in row:
                prompt += f'\n{choice}: {row[choice]}'
        image = row['image']
        try:
            pil_image = decode_base64_image_to_pil(image)
            byte_io = io.BytesIO()
            pil_image.save(byte_io, format='PNG')
            bytes_image = byte_io.getvalue()
        except Exception as e:
            print(f'Error in {i}th row: {e}')
            continue
        images = [{'bytes': bytes_image}]
        answer = row['answer']
        new_row = {
            'data_source': 'hiyouga/geometry3k',
            "prompt": [
                    {
                        "role": "user",
                        "content": '<image>'+prompt,
                    }
                ],
            "images": images,
            "ability": "general",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": split,
                "index": i,
                "answer": answer,
                "question": prompt,
            },
        }
        if split == 'train':
            train_results.append(new_row)
        else:
            test_results.append(new_row)
    train_df = pd.DataFrame(train_results)
    test_df = pd.DataFrame(test_results)
    train_df.to_parquet(f'{output_root}/train.parquet')
    test_df.to_parquet(f'{output_root}/test.parquet')