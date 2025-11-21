from datakit.utils.files import read_jsonl_file, dump_list_to_jsonl_file
from datakit.utils.mp import multi_process_with_append


def filter_single_item(item):
    image_key = list(item['base64_image'].keys())[0]
    base64_image = item['base64_image'][image_key]
    if base64_image is None:
        return None
    conversations = item['conversations']
    for i in range(len(conversations)//2):
        answer_index = i*2+1
        answer = conversations[answer_index]['text']
        if len(answer) > 30:
            return item
    return None


if __name__ == '__main__':
    ori_file = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/sharegpt4v/data/grammar_correct/sharegpt4v.jsonl'
    output_file = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/sharegpt4v/data/grammar_correct_caption/sharegpt4v.jsonl'
    data = read_jsonl_file(ori_file)
    results = multi_process_with_append(filter_single_item, data, 64)
    dump_list_to_jsonl_file(output_file, results)

