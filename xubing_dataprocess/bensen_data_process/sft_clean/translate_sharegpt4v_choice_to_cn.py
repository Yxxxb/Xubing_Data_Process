from datakit import vLLMWrapper
from datakit.utils.files import read_jsonl_file, dump_list_to_jsonl_file
from datakit.utils.distributed import dist_split_files, get_distributed_env
import os
from tqdm import tqdm


with open('points-1.2/prompts/translate_en2cn.txt') as f:
    PROMPT = f.read()

model_path = '/mnt/cephfs/bensenliu/wfs/weights/nlp/Qwen2.5-32B-Instruct'
model = vLLMWrapper(model_path, 8, max_tokens=512)

if __name__ == '__main__':
    data_file = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/sharegpt4v_choice/data/grammar_correct/sharegpt4v_choice.jsonl'
    data = read_jsonl_file(data_file)
    data_cur_rank = dist_split_files(data)
    world_size, rank, local_rank = get_distributed_env()
    output_file = f'/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/sharegpt4v_choice_cn/data/grammar_correct/sharegpt4v_choice_cn_{rank}.jsonl'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results = []
    for item in tqdm(data_cur_rank):
        conversations = item['conversations']
        for i in range(len(conversations)):
            if i % 2 == 0:
                text = conversations[i]['text']
                prompt = PROMPT + text + '\n### 翻译结果\n'
                cn_text = model.generate(prompt, use_tqdm=False)
                conversations[i]['text'] = cn_text
        item['conversations'] = conversations
        results.append(item)
    dump_list_to_jsonl_file(output_file, results)
    