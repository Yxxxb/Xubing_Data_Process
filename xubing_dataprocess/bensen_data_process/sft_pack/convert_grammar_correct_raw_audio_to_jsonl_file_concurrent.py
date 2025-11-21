from datakit.utils.files import (read_jsonl_file,
                                 filterout_repeat_images_for_mmq,
                                 dump_list_to_jsonl_file,
                                 find_all_files)
from datakit.utils.distributed import (dist_split_files,
                                       get_distributed_env)
from datakit.utils.image import (check_image_integrity,
                                 decode_base64_image_to_np,
                                 encode_np_to_base64_image)
from data_distribution import data_distribution
from video_data_distribution import video_data_distribution
from datakit.utils.mp import multi_process_with_append
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.image_transforms import resize
from transformers.image_utils import ChannelDimension
from tqdm import tqdm
import os
from pprint import pprint
from datetime import datetime
import pytz


is_video = False
SUBSET = 'grammar_correct'
max_pixels = 14 * 14 * 4 * 1280
include_datasets = ['coco', 'allava_cap', 'gpt4v-cn', 'lvis_instruct4v_cap_cn', 'ChartQA', 'sharegpt4v_ref', 'alpaca-gpt4', 'belle_cn_0.5m', 'belle_cn_1m', 'mini-gemini', 'math', '500k-atlas-math', 'MetaMathQA', 'sharegpt4v_sharegpt', 'MathInstruct', 'orca-math-word-problems-200k', 'openhermes2.5', 'lima', 'belle_cn_2m', 'stvqa', 'ESTVQA', 'Chinese-OCR', 'express_waybill_cn', 'hme100k', 'DocOCR-render', 'bank_card_cn', 'invoices-and-receipts_ocr_v1_image_translation', 'ride_sharing_cn', 'markdown2image', 'docvqa', 'toll_invoice_cn', 'medical_bill_cn', 'bus_ticket_cn', 'steamer_ticket_cn', 'icdar_2015', 'handwritten_cn', 'duty-paid-proof-cn', 'poie', 'value-added-tax-invoice-cn', 'taxi_ticket_cn', 'LaTeX_OCR_column_rename_full', 'tabwp_cot', 'sharegpt4v_textcaps', 'vat_invoice_roll_cn', 'souyisou-ocr', 'welm_pdf', 'UniMER', 'k12-print', 'customs_declaration_cn', 'quota_invoice_cn', 'LaTeX_OCR', 'llavar', 'railway_ticket_cn', 'aiplane-itinerary-cn', 'PubTabNet', 'welm_pdf_markdown', 'Chinese-Card-OCR', 'textvqa', 'invoice-mix-up-cn', 'mapqa', 'infovqa', 'memes-500', 'clevr_math_5w', 'iconqa_choose_txt', 'geoqa+', 'super_clever', 'MathV360K', 'iconqa_fill_blank', 'geo3k', 'mm_ai2d', 'tqa', 'scienceqa', 'kvqa', 'Mantis-Instruct-multi_vqa', 'gpt4o-complex-20240809-en', 'Mantis-Instruct-spot-the-diff', 'Mantis-Instruct-dreamsim', 'sharegpt4v_choice', 'Mantis-Instruct-contrastive_caption', 'Mantis-Instruct-birds-to-words', 'sharegpt4v_choice_cn', 'sharegpt4v_qa', 'Mantis-Instruct-nlvr2', 'sharegpt4v_llava158k', 'sharegpt4v_qa_cn', 'sft-enhanced', 'Mantis-Instruct-iconqa', 'vsr', 'Mantis-Instruct-lrv_multi']
if is_video:
    root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category-video'
    output_root = root
    data_structure = video_data_distribution
else:
    root = '/mnt/cephfs/neuronzhang/Data/VisualData/mm2audio'
    output_root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category'
    data_structure = data_distribution
points = []
for category, dataset_names in data_structure.items():
    for dataset_name in dataset_names:
        if dataset_name not in include_datasets:
            continue
        points.append(f'{category}/{dataset_name}')


def add_dummy_image(data, data_file):
    dummy_image = {
        'id': data_file,
        'type': 'image',
        'content': 'This is the content of dummy image'
    }
    data.append(dummy_image)
    return data


def pack_single_row(item):
    base64_image_dict = item['base64_image']
    conversations = item['conversations']
    keys = list(base64_image_dict.keys())
    image_id = item['id']
    if len(keys) == 0 or base64_image_dict[keys[0]] is None:
        conversation_template = {
            'type': 'conversation',
            'id': image_id,
            'conversations': conversations
        }
        return [conversation_template]
    if conversations[1]['text'] is None or '<|image_pad|>' in conversations[1]['text']:
        print(f'Skip {image_id} because of <|image_pad|> in text')
        return None
    image_templates = []
    image_text = ''
    seq_lens = dict()
    for image_name, base64_image in base64_image_dict.items():
        if not check_image_integrity(base64_image):
            return None
        try:
            np_image = decode_base64_image_to_np(base64_image)
            height, width = np_image.shape[:2]
            if height < 28 or width < 28:
                # resize the smallest side to 28, keeping aspect ratio
                if height < width:
                    new_height = 28
                    new_width = int(28 * width / height)
                else:
                    new_width = 28
                    new_height = int(28 * height / width)
            else:
                new_height = height
                new_width = width
            resized_height, resized_width = smart_resize(new_height,
                                                         new_width,
                                                         max_pixels=max_pixels)
            image = resize(np_image, size=(resized_height, resized_width),
                           resample=3, input_data_format=ChannelDimension.LAST)
            if image.shape[-1] == 1:
                image = image.repeat(3, axis=-1)
            base64_image = encode_np_to_base64_image(image)
            assert resized_height % 28 == 0 and resized_width % 28 == 0
            seq_len = int(resized_width / 28 * resized_height / 28)
            assert seq_len > 0
            utc_now = datetime.now(pytz.utc)
            reformat_image_name = image_name + utc_now.strftime('%Y-%m-%d %H:%M:%S.%f %Z%z') # noqa
            seq_lens[reformat_image_name] = seq_len
        except Exception as e:
            print(e)
            return None
        image_text += f'<img>{reformat_image_name}</img>'
        image_template = {
            'id': reformat_image_name,
            'type': 'image',
            'content': base64_image
        }
        image_templates.append(image_template)
    if 'image_is_pack' not in item or not item['image_is_pack']:
        conversations[0]['text'] = image_text + conversations[0]['text']
    else:
        print(f'Skip {image_id} because of image_is_pack')
        return None
    conversation_template = {
        'type': 'conversation',
        'seq_lens': seq_lens,
        'id': image_id,
        'conversations': conversations
    }
    return [conversation_template] + image_templates


def pack_single_jsonl_file(files):
    input_file, output_file = files
    data = read_jsonl_file(input_file)
    results = []
    for item in tqdm(data):
        result = pack_single_row(item)
        if result is not None:
            results.extend(result)
    results = filterout_repeat_images_for_mmq(results)
    dump_list_to_jsonl_file(output_file, results)


if __name__ == '__main__':
    filenames_cur_rank = dist_split_files(points)
    world_size, rank, _ = get_distributed_env()
    pprint(
        f'Current rank {os.environ["RANK"]} processing {filenames_cur_rank}')
    success_files = []
    failed_files = []
    for filename in tqdm(filenames_cur_rank):
        sub_filename = filename.split('/')[-1]
        print(f'Processing {filename}')
        input_folder = os.path.join(
            root, filename, 'data', SUBSET)
        input_files = find_all_files(input_folder, 'jsonl')
        if len(input_files) == 0:
            print(f'Skip {filename} because of no input files')
            continue
        output_folder = os.path.join(
            output_root, filename, f'jsonl/qwen2vl-audio-{SUBSET}')
        output_files = [os.path.join(output_folder, os.path.basename(f))
                        for f in input_files]
        os.makedirs(output_folder, exist_ok=True)
        files = list(zip(input_files, output_files))
        try:
            multi_process_with_append(pack_single_jsonl_file, files,
                                      num_workers=32)
            success_files.append(filename)
        except Exception as e:
            print(f'Error in {filename}: {e}')
            failed_files.append(filename)
    print(f'Success files: {success_files}')
    print(f'Failed files: {failed_files}')
