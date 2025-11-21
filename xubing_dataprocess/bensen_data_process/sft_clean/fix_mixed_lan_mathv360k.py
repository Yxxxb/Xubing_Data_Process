from datakit.utils.files import read_jsonl_file, dump_list_to_jsonl_file
from datakit.utils.mp import multi_process_with_append
import os


PATTERN_MAPPING = {
    'Please answer the question and provide the final answer at the end.': '请回答问题并在最后提供最终答案。',
    'Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.': '请回答问题并在最后提供正确的选项字母，例如 A、B、C、D。',
    'Question:': '问题:',
    'Choices:': '选项:',
    
}


def fix_single_question(item):
    conversations = item['conversations']
    question = conversations[0]['text']
    for pattern_key, pattern_value in PATTERN_MAPPING.items():
        question = question.replace(pattern_key, pattern_value)
    item['conversations'][0]['text'] = question
    return item


if __name__ == '__main__':
    ori_file = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/MathV360K/data/grammar_correct_mixed_lan/MathV360K.jsonl'
    output_file = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/MathV360K/data/grammar_correct_mixed_lan_fix/MathV360K.jsonl'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)   
    data = read_jsonl_file(ori_file)
    results = multi_process_with_append(fix_single_question, data, num_workers=64)
    dump_list_to_jsonl_file(output_file, results)

    

