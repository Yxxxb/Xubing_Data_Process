from datakit.utils.files import (read_jsonl_file,
                                 filterout_repeat_images_for_mmq,
                                 dump_list_to_jsonl_file,
                                 find_all_files)
from datakit.utils.distributed import (dist_split_files)
from datakit.utils.image import (check_image_integrity,
                                 decode_base64_image_to_np,
                                 encode_np_to_base64_image)
from datakit.utils.mp import multi_process_with_append
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.image_transforms import resize
from transformers.image_utils import ChannelDimension
import os
from datetime import datetime
import pytz


max_pixels = 2048 * 28 * 28


def pack_single_row(item):
    base64_image_dict = item['base64_image']
    conversations = item['conversations']
    if len(conversations[-1]['text']) < 100:
        return None
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
    for item in data:
        result = pack_single_row(item)
        if result is not None:
            results.extend(result)
    results = filterout_repeat_images_for_mmq(results)
    dump_list_to_jsonl_file(output_file, results)


if __name__ == '__main__':
    folders = [
        '/mnt/cephfs/bensenliu/wfs/vlmdatasets/pt/welm-pdf/k12_1684_jsonl_ocr',
        '/mnt/cephfs/bensenliu/wfs/vlmdatasets/pt/welm-pdf/606中文-jsonl_ocr',
        '/mnt/cephfs/bensenliu/wfs/vlmdatasets/pt/welm-pdf/chinese_jsonl_ocr',
        '/mnt/cephfs/bensenliu/wfs/vlmdatasets/pt/welm-pdf/1701英语-jsonl_ocr'
    ]
    all_jsonls = []
    for folder in folders:
        all_jsonls.extend(find_all_files(folder, 'jsonl'))
    output_root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/doc_ocr/pt/data/20250313e1'
    output_jsonls = [os.path.join(output_root, f'{index}.jsonl')
                     for index in range(len(all_jsonls))]
    input_data = list(zip(all_jsonls, output_jsonls))
    input_data_cur_rank = dist_split_files(input_data)
    multi_process_with_append(pack_single_jsonl_file, input_data_cur_rank,
                              num_workers=512)
    print('Done')