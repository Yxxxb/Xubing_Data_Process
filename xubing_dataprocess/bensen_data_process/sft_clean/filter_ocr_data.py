from datakit.utils.files import read_jsonl_file, dump_list_to_jsonl_file
from datakit.utils.mp import multi_process_with_append
from tqdm import tqdm


datasets = [
    'Chinese-Card-OCR',
    'Chinese-OCR',
    'aiplane-itinerary-cn',
    'bank_card_cn',
    'boarding_pass_cn',
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
    'value-added-tax-invoice-cn',
    'vat_invoice_roll_cn'
]


def process_single_item(item):
    answer = item['conversations'][1]['text']
    if '如下：' in answer:
        answer = answer.split('如下：')[1]
        item['conversations'][1]['text'] = answer
    elif '文字：' in answer:
        answer = answer.split('文字：')[1]
        item['conversations'][1]['text'] = answer
    elif '图片中的文字' in answer:
        return None
    elif 'markdown' in answer:
        return None
    return item


if __name__ == '__main__':
    root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/OCR'
    for dataset in tqdm(datasets):
        data_path = f'{root}/{dataset}/data/grammar_correct/{dataset}.jsonl'
        data = read_jsonl_file(data_path)
        results = multi_process_with_append(process_single_item, data, 128)
        print(f'original data size: {len(data)}, filtered data size: {len(results)}')
        dump_list_to_jsonl_file(data_path, results)
