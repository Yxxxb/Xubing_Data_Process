from datakit.utils.files import read_jsonl_file, dump_list_to_jsonl_file
from datakit.utils.distributed import dist_split_files, get_distributed_env
from datakit import vLLMWrapper
from tqdm import tqdm
import re
import os


with open('points-1.2/prompts/translate_sharegpt4vqa_en2cn.txt', encoding='utf-8') as f:
    PROMPT = f.read()

model_path = '/mnt/cephfs/bensenliu/wfs/weights/nlp/Qwen2.5-32B-Instruct'
model = vLLMWrapper(model_path, 8, max_tokens=512)


def extract_question_answer_pair(response):
    question = re.search(r'question:(.*?)\nanswer:', response, re.DOTALL)
    if question:
        question = question.group(1).strip()
    else:
        return None, None
    answer = re.search(r'answer:(.*)', response, re.DOTALL)
    if answer:
        answer = answer.group(1).strip()
    else:
        return None, None
    return question, answer


if __name__ == '__main__':
    _, rank, local_rank = get_distributed_env()
    data_file = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/sharegpt4v_qa/data/grammar_correct/sharegpt4v_qa.jsonl'
    output_file = f'/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/sharegpt4v_qa_cn/data/grammar_correct/sharegpt4v_qa_{rank}.jsonl'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    world_size, rank, local_rank = get_distributed_env()
    data = read_jsonl_file(data_file)
    data_cur_rank = dist_split_files(data)
    results = []
    for item in tqdm(data_cur_rank):
        conversations = item['conversations']
        new_conversations = []
        for i in range(len(conversations)//2):
            question = conversations[2*i]['text']
            answer = conversations[2*i+1]['text']
            prompt = f'question: {question}\nanswer: {answer}\n'
            prompt = PROMPT + prompt + '### 翻译结果'
            cn_text = model.generate(prompt, use_tqdm=False).strip()
            question_cn, answer_cn = extract_question_answer_pair(cn_text)
            if question_cn and answer_cn:
                new_conversations.extend([
                    {
                        'role': 'user',
                        'text': question_cn
                    },
                    {
                        'role': 'assistant',
                        'text': answer_cn
                    }
                ])
        item['conversations'] = new_conversations
        results.append(item)
    dump_list_to_jsonl_file(output_file, results)
            

            

