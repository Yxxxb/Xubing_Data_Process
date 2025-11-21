from datakit.utils.files import read_jsonl_file, dump_list_to_jsonl_file
from datakit.utils.mp import multi_process_with_append


def process_single_line(line):
    conversations = line['conversations']
    question = conversations[0]['text']
    if '<image>' in question:
        question = question.replace('\n<image>', ' ')
        question = question.replace('<image>\n', ' ')
        question = question.strip()
        conversations[0]['text'] = question
        line['conversations'] = conversations
    return line


if __name__ == '__main__':
    jsonl_file = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/sharegpt4v/data/grammar_correct/sharegpt4v.jsonl'
    data = read_jsonl_file(jsonl_file)
    results = multi_process_with_append(process_single_line, data, num_workers=32)
    dump_list_to_jsonl_file(jsonl_file, results)
    
