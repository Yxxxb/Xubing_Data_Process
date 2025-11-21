from datakit.utils.files import find_all_files, read_jsonl_file, dump_list_to_jsonl_file
from tqdm import tqdm
import os
from datakit.utils.mp import multi_process_with_extend


def filter_single_file(jsonl_file):
    data = read_jsonl_file(jsonl_file)
    results = []
    for item in tqdm(data):
        conversations = item['conversations']
        new_conversations = []
        for i in range(len(conversations)//2):
            text = conversations[2*i+1]['text']
            if '注' in text or '翻译' in text:
                continue
            new_conversations.append(conversations[2*i])
            new_conversations.append(conversations[2*i+1])
        if len(new_conversations) > 0:
            item['conversations'] = new_conversations
            results.append(item)
    return results


if __name__ == '__main__':
    data_root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/sharegpt4v_qa_cn/data/grammar_correct'
    correct_file = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/sharegpt4v_qa_cn/data/grammar_correct_translated_correct/sharegpt4v_qa_cn.jsonl'
    os.makedirs(os.path.dirname(correct_file), exist_ok=True)
    jsonl_files = find_all_files(data_root, 'jsonl')
    results = multi_process_with_extend(filter_single_file, jsonl_files, 
                                        num_workers=len(jsonl_files))
    dump_list_to_jsonl_file(correct_file, results)
