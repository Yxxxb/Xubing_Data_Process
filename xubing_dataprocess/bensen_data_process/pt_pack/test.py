from datakit.utils.files import read_jsonl_file, dump_list_to_jsonl_file
from datakit.utils.mp import multi_process_with_append



def pack_single_item(item):
    conversations = item['conversations']
    