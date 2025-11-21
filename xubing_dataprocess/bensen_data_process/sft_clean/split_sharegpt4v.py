from datakit.utils.files import read_jsonl_file, dump_list_to_jsonl_file
from tqdm import tqdm
import os


def construct_conversations(conversations):
    conversation_str = ''
    for conversation in conversations:
        role = conversation['role']
        text = conversation['text']
        conversation_str += f'{role}: {text}\n'
    return conversation_str.strip()


QA_PATTERN = "Answer the question using a single word or phrase."
CHOICE_PATTERN = "Answer with the option's letter from the given choices directly."
REF_PATTERN = "Please provide the bounding box coordinate of the region this sentence describes"
TEXTCAPS_PATTERN = "Provide a one-sentence caption for the provided image."


sharegpt4v_qa_path = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/sharegpt4v_qa/data/grammar_correct'
sharegpt4v_choice_path = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/sharegpt4v_choice/data/grammar_correct'
sharegpt4v_ref_path = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/sharegpt4v_ref/data/grammar_correct'
sharegpt4v_textcaps_path = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/sharegpt4v_textcaps/data/grammar_correct'
sharegpt4v_llava158k_path = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/sharegpt4v_llava158k/data/grammar_correct'
sharegpt4v_sharegpt_path = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/sharegpt4v_sharegpt/data/grammar_correct'
sharegpt4v_paths = [sharegpt4v_qa_path, sharegpt4v_choice_path, 
                    sharegpt4v_ref_path, sharegpt4v_textcaps_path, 
                    sharegpt4v_llava158k_path, sharegpt4v_sharegpt_path]

qa_data = []
choice_data = []
ref_data = []
textcaps_data = []
llava158k_data = []
sharegpt_data = []


if __name__ == '__main__':
    for sharegpt4v_path in sharegpt4v_paths:
        os.makedirs(os.path.dirname(sharegpt4v_path), exist_ok=True)
    data_path = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2/sharegpt4v/data/grammar_correct/sharegpt4v.jsonl'
    data = read_jsonl_file(data_path)
    for item in tqdm(data):
        image_key = list(item['base64_image'].keys())[0]
        base64_image = item['base64_image'][image_key]
        if base64_image is None:
            # sharegpt
            sharegpt_data.append(item)
            continue
        conversations = item['conversations']
        conversations_str = construct_conversations(conversations)
        if QA_PATTERN in conversations_str: 
            qa_data.append(item)
        elif CHOICE_PATTERN in conversations_str:
            choice_data.append(item)
        elif REF_PATTERN in conversations_str:
            ref_data.append(item)
        elif TEXTCAPS_PATTERN in conversations_str:
            textcaps_data.append(item)
        else:
            llava158k_data.append(item)
    print(f'QA: {len(qa_data)}')
    dump_list_to_jsonl_file(sharegpt4v_qa_path, qa_data)
    print(f'Choice: {len(choice_data)}')
    dump_list_to_jsonl_file(sharegpt4v_choice_path, choice_data)
    print(f'Ref: {len(ref_data)}')
    dump_list_to_jsonl_file(sharegpt4v_ref_path, ref_data)
    print(f'Textcaps: {len(textcaps_data)}')
    dump_list_to_jsonl_file(sharegpt4v_textcaps_path, textcaps_data)
    print(f'LLAVA158k: {len(llava158k_data)}')
    dump_list_to_jsonl_file(sharegpt4v_llava158k_path, llava158k_data)
    print(f'ShareGPT: {len(sharegpt_data)}')
    dump_list_to_jsonl_file(sharegpt4v_sharegpt_path, sharegpt_data)
    
        
