from datakit.utils.files import read_jsonl_file, dump_list_to_jsonl_file
from datakit.utils.mp import multi_process_with_append
import os


PATTERN_MAPPING = {
    "Answer with the option's letter from the given choices directly.": '直接用给定选项中的字母回答。',
    "Answer the question using a single word or phrase.": "用一个单词或短语回答问题。"
    
}


def fix_single_question(item):
    conversations = item['conversations']
    question = conversations[0]['text']
    for pattern_key, pattern_value in PATTERN_MAPPING.items():
        question = question.replace(pattern_key, pattern_value)
    item['conversations'][0]['text'] = question
    return item


if __name__ == '__main__':
    ori_file = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/geoqa+/data/grammar_correct_mixed_lan/geoqa+.jsonl'
    output_file = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/geoqa+/data/grammar_correct_mixed_lan_fix/geoqa+.jsonl'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)   
    data = read_jsonl_file(ori_file)
    results = multi_process_with_append(fix_single_question, data, num_workers=64)
    dump_list_to_jsonl_file(output_file, results)

    

