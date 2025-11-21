import subprocess
import os
from tqdm import tqdm
from datakit.utils.distributed import (dist_split_files,
                                       get_distributed_env)
from datakit.utils.files import (find_all_files)
from data_distribution import data_distribution
from video_data_distribution import video_data_distribution
from datakit.utils.distributed import (barrier_all_processes, 
                                       gpu_utilization,
                                       kill_process)


is_video = False
SUBSET = 'grammar_correct'
include_datasets = [
    'Chinese-Card-OCR',
    'Chinese-OCR',
    'sharegpt4v_textcaps',
    'stvqa',
    'textvqa',
    'k12-print',
    'handwritten_cn',
    'textvqa',
    'k12-print',
    'handwritten_cn',
    'ESTVQA',
    'value-added-tax-invoice-cn',
    'aiplane-itinerary-cn',
    'bank_card_cn',
    'bus_ticket_cn',
    'customs_declaration_cn',
    'duty-paid-proof-cn',
    'express_waybill_cn',
    'invoice-mix-up-cn',
    'medical_bill_cn',
    'quota_invoice_cn',
    'railway_ticket_cn',
    'ride_sharing_cn',
    'steamer_ticket_cn',
    'taxi_ticket_cn',
    'toll_invoice_cn',
    'vat_invoice_roll_cn',
    'invoices-and-receipts_ocr_v1_image_translation',
    'docvqa',
    'allava_cap',
    'gpt4v-cn',
    'lvis_instruct4v_cap_cn',
    'ChartQA',
    'infovqa',
    'mapqa',
    'gpt4o-complex-20240809-en',
    'kvqa',
    'sft-enhanced',
    'sharegpt4v_choice',
    'sharegpt4v_choice_cn',
    'sharegpt4v_llava158k',
    'sharegpt4v_qa',
    'sharegpt4v_qa_cn',
    'vsr'
]
if is_video:
    root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category-video'
    data_structure = video_data_distribution
else:
    root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category'
    data_structure = data_distribution
points = []
for category, dataset_names in data_structure.items():
    for dataset_name in dataset_names:
        if dataset_name not in include_datasets:
            continue
        points.append(f'{category}/{dataset_name}')


if __name__ == '__main__':
    sub_folder = f'recordio/qwen2vl-audio-grammar_correct-qwen-2-5'
    recordios = []
    for point in points:
        folder = os.path.join(root, point, sub_folder)
        recordios.extend(find_all_files(folder, 'data.recordio'))
    print(' '.join(recordios))

