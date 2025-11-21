from datakit.utils.files import (read_jsonl_file,
                                 filterout_repeat_images_for_mmq,
                                 dump_list_to_jsonl_file)
from datakit.utils.distributed import (dist_split_files,
                                       get_distributed_env)
from datakit.utils.image import (check_image_integrity,
                                 decode_base64_image_to_np,
                                 encode_np_to_base64_image)
from data_distribution import data_distribution
from video_data_distribution import video_data_distribution
from datakit.utils.mp import multi_process_with_extend
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.image_transforms import resize
from transformers.image_utils import ChannelDimension
from tqdm import tqdm
import os
from pprint import pprint


is_video = False
SUBSET = 'grammar_correct_remove_space'
max_pixels = 14 * 14 * 4 * 2048
include_datasets = [
    'hme100k',
    'LaTeX_OCR',
    'LaTeX_OCR_column_rename_full',
    'UniMER'
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
    for conversation in conversations:
        text = conversation['text']
        if '<img>' in text:
            print(f'Skip {image_id} because of <img> in text')
            return None
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
            seq_lens[image_name] = seq_len
        except Exception as e:
            print(e)
            return None
        image_text += f'<img>{image_name}</img>'
        image_template = {
            'id': image_name,
            'type': 'image',
            'content': base64_image
        }
        image_templates.append(image_template)
    if 'image_is_pack' not in item or not item['image_is_pack']:
        conversations[0]['text'] = image_text + conversations[0]['text']
    conversation_template = {
        'type': 'conversation',
        'seq_lens': seq_lens,
        'id': image_id,
        'conversations': conversations
    }
    return [conversation_template] + image_templates


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
        cur_file = os.path.join(
            root, filename, 'data', SUBSET,
            f'{sub_filename}.jsonl')
        if not os.path.exists(cur_file):
            print(f'{cur_file} does not exist')
            continue
        data = read_jsonl_file(cur_file)
        if len(data) == 0:
            print(f'{cur_file} is empty')
            continue
        output_file = os.path.join(
            root, filename, f'jsonl/qwen2vl-{SUBSET}/{sub_filename}.jsonl')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        try:
            data = multi_process_with_extend(
                pack_single_row, data, num_workers=512)
            data = filterout_repeat_images_for_mmq(data)
            data = add_dummy_image(data, cur_file)
            dump_list_to_jsonl_file(output_file, data)
            success_files.append(filename)
        except Exception as e:
            print(f'Error in {filename}: {e}')
            failed_files.append(filename)
    print(f'Success files: {success_files}')
    print(f'Failed files: {failed_files}')
