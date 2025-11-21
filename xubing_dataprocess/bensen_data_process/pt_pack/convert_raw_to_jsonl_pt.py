from datakit.utils.files import (find_all_files,
                                 dump_list_to_jsonl_file,
                                 filterout_repeat_images_for_mmq,
                                 read_jsonl_file)
from datakit.utils.image import (check_image_integrity,
                                 decode_base64_image_to_np,
                                 encode_np_to_base64_image)
from datakit.utils.distributed import dist_split_files
from datakit.utils.mp import multi_process_with_extend
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.image_transforms import resize
from transformers.image_utils import ChannelDimension
from tqdm import tqdm
import os


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
                                                         new_width)
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
            print(f'Error in {image_name}: {e}')
            return None
        image_text += f'<img>{image_name}</img>'
        image_template = {
            'id': image_name,
            'type': 'image',
            'content': base64_image
        }
        image_templates.append(image_template)
    conversation_template = {
        'type': 'conversation',
        'seq_lens': seq_lens,
        'id': image_id,
        'conversations': conversations
    }
    return [conversation_template] + image_templates


if __name__ == '__main__':
    root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/pt/obelics/data/grammar_correct_1M'
    output_root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/pt/obelics/jsonl/qwen2vl-grammar_correct_1M'
    os.makedirs(output_root, exist_ok=True)
    all_files = find_all_files(root, 'jsonl')
    files_cur_rank = dist_split_files(all_files)
    for file in tqdm(files_cur_rank):
        filename = os.path.basename(file)
        foldername = file.split('/')[-2]
        filename = f'{foldername}_{filename}'
        output_file = os.path.join(output_root, filename)
        try:
            rows = read_jsonl_file(file)
        except Exception as e:
            print(f'Error reading file {file}: {e}')
            continue
        results = multi_process_with_extend(
            pack_single_row, rows, num_workers=128)
        results = filterout_repeat_images_for_mmq(results)
        dump_list_to_jsonl_file(output_file, results)
    print('Done!')
