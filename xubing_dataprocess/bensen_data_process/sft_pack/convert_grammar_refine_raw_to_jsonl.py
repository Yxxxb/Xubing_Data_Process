from datakit.utils.files import (find_all_files,
                                 read_jsonl_file,
                                 filterout_repeat_images_for_mmq,
                                 dump_list_to_jsonl_file)
from datakit.utils.distributed import (dist_split_files,
                                       get_distributed_env)
from datakit.utils.image import (check_image_integrity,
                                 decode_base64_image_to_np,
                                 encode_np_to_base64_image)
from datakit.utils.mp import multi_process_with_extend
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.image_transforms import resize
from transformers.image_utils import ChannelDimension
from tqdm import tqdm
import os
from pprint import pprint


points_sft = [
    'mm_ai2d',
    'docvqa',
    'dvqa',
    'geoqa+',
    'allava_cap',
    'iconqa_choose_txt',
    'iconqa_fill_blank',
    'infovqa',
    'kvqa',
    'gpt4v',
    'llavar',
    'scienceqa',
    'sharegpt4v',
    'stvqa',
    'super_clever',
    'textvqa',
    'tqa',
    'vsr',
    'icdar_2015',
    'lima',
    'alpaca-gpt4',
    'openhermes2.5',
    'mini-gemini',
    'hme100k',
    'tabwp_cot',
    'geo3k',
    'clevr_math_5w',
    'poie',
    'lvis_instruct4v_cap',
    'MetaMathQA',
    'MathInstruct',
    'orca-math-word-problems-200k',
    'math',
    '500k-atlas-math',
    'gpt4o-complex-20240809-en',
    'MathV360K'
]


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
    image_templates = []
    image_text = ''
    seq_lens = dict()
    for image_name, base64_image in base64_image_dict.items():
        if not check_image_integrity(base64_image):
            return None
        try:
            np_image = decode_base64_image_to_np(base64_image)
            height, width = np_image.shape[:2]
            resized_height, resized_width = smart_resize(height,
                                                         width)
            image = resize(np_image, size=(resized_height, resized_width),
                           resample=3, input_data_format=ChannelDimension.LAST)
            if image.shape[-1] == 1:
                image = image.repeat(3, axis=-1)
            base64_image = encode_np_to_base64_image(image)
            seq_len = int(resized_width / 28 * resized_height / 28)
            seq_lens[image_name] = seq_len
        except Exception as e:
            print(f'Error in {image_name}: {e}')
            return None
        image_text += f'<img>{image_name}</img>'
        image_template = {
            'id': image_name,
            'type': 'image',
            'content': base64_image
        }
        image_templates.append(image_template)

    conversations[0]['text'] = image_text + conversations[0]['text']
    conversation_template = {
        'type': 'conversation',
        'seq_lens': seq_lens,
        'id': image_id,
        'conversations': conversations
    }
    return [conversation_template] + image_templates


if __name__ == '__main__':
    filenames_cur_rank = dist_split_files(points_sft)
    world_size, rank, _ = get_distributed_env()
    pprint(
        f'Current rank {os.environ["RANK"]} processing {filenames_cur_rank}')
    root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-pointsv1.2'
    success_files = []
    failed_files = []
    for filename in tqdm(filenames_cur_rank):
        print(f'Processing {filename}')
        cur_folder = os.path.join(
            root, filename, 'data', 'grammar_error_refined')
        output_folder = os.path.join(
            root, filename, 'jsonl/qwen2vl-grammar-error-refine')
        os.makedirs(output_folder, exist_ok=True)
        target_files = find_all_files(cur_folder, 'jsonl')
        if len(target_files) == 0:
            print(f'Skip {filename} because no jsonl files found')
            continue
        try:
            for target_file in target_files:
                target_file_name = os.path.basename(target_file)
                output_file = os.path.join(output_folder, target_file_name)
                data = read_jsonl_file(target_file)
                data = multi_process_with_extend(
                    pack_single_row, data, num_workers=64)
                data = filterout_repeat_images_for_mmq(data)
                data = add_dummy_image(data, target_file_name)
                dump_list_to_jsonl_file(output_file, data)
            success_files.append(filename)
        except Exception as e:
            print(f'Error in {filename}: {e}')
            failed_files.append(target_file)
    print(f'Success files: {success_files}')
    print(f'Failed files: {failed_files}')
