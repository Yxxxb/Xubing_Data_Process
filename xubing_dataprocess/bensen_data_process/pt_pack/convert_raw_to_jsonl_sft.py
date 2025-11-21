from datakit.utils.files import (find_all_files,
                                 read_jsonl_file,
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


MAX_PIXELS = 14 * 14 * 4 * 160

def pack_single_row(item):
    base64_image_dict = item['base64_images'] if 'base64_images' in item else item['base64_image']
    conversations = item['conversations']
    keys = list(base64_image_dict.keys())
    image_id = item['id']
    if len(keys) == 0 or base64_image_dict[keys[0]] is None:
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
            resized_height, resized_width = smart_resize(height,
                                                         width,
                                                         max_pixels=MAX_PIXELS)
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
    root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category-video/CAPTION/OpenVid-1M/data/80w'
    output_root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category-video/CAPTION/OpenVid-1M/jsonl/qwen2vl-80w'
    all_jsonls = find_all_files(root, 'jsonl')
    jsonls_cur_rank = dist_split_files(all_jsonls)
    world_size, rank, _ = get_distributed_env()
    pprint(
        f'Current rank {os.environ["RANK"]} processing {len(jsonls_cur_rank)} files')
    success_files = []
    failed_files = []
    for jsonl in tqdm(jsonls_cur_rank):
        sub_folder_name = os.path.basename(os.path.dirname(jsonl))
        jsonl_name = os.path.basename(jsonl)
        output_file = os.path.join(output_root, sub_folder_name, jsonl_name)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        try:
            data = read_jsonl_file(jsonl)
            data = multi_process_with_extend(
                pack_single_row, data, num_workers=128)
            dump_list_to_jsonl_file(output_file, data)
            success_files.append(jsonl)
        except Exception as e:
            print(f'Error in {jsonl}: {e}')
            failed_files.append(jsonl)
    if len(failed_files) > 0:
        print(f'Failed files: {failed_files}')
    else:
        print('All files processed successfully')
