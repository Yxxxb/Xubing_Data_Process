from datakit.utils.files import (read_jsonl_file,
                                 filterout_repeat_images_for_mmq,
                                 dump_list_to_jsonl_file)
from datakit.utils.distributed import (dist_split_files)
from datakit.utils.image import (check_image_integrity,
                                 decode_base64_image_to_pil)
from datakit.utils.files import find_all_files
from datakit.utils.mp import multi_process_with_extend
from transformers.models.qwen2_vl import Qwen2VLImageProcessor
import numpy as np
from datetime import datetime
import pytz
import os
from tqdm import tqdm


MAX_PIXELS = 14 * 14 * 4 * 160
image_processor = Qwen2VLImageProcessor.from_pretrained('/mnt/cephfs/bensenliu/wfs/weights/mm/opensource/Qwen2-VL-7B-Instruct')


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
    if '<|image_pad|>' in conversations[1]['text']:
        print(f'Skip {image_id} because of <|image_pad|> in text')
        return None
    pil_images = []
    image_templates = []
    image_text = ''
    try:
        for base64_name, base64_image in base64_image_dict.items():
            if not check_image_integrity(base64_image):
                return None
            pil_images.append(decode_base64_image_to_pil(base64_image))
            utc_now = datetime.now(pytz.utc)
            base64_name = base64_name + utc_now.strftime('%Y-%m-%d %H:%M:%S.%f %Z%z')
            image_text += f'<img>{base64_name}</img>'
            image_template = {
                'id': base64_name,
                'type': 'image',
                'content': base64_image
            }
            image_templates.append(image_template)
        output = image_processor(images=None, videos=pil_images, max_pixels=MAX_PIXELS)
    except Exception as e:
        print(e)
        return None
    video_grid_thw = output['video_grid_thw']
    seq_len = int(np.prod(video_grid_thw)) // 4
    conversations[0]['text'] = image_text + conversations[0]['text']
    conversation_template = {
        'type': 'conversation',
        'seq_lens': {image_id: seq_len},
        'max_pixels': MAX_PIXELS,
        'is_video': True,
        'id': image_id,
        'conversations': conversations
    }
    return [conversation_template] + image_templates


if __name__ == '__main__':
    raw_root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category-video/QA/LLaVA-Video-178K-refine/data/grammar_correct_1fps_64maxframes'
    jsonl_root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category-video/QA/LLaVA-Video-178K-refine/jsonl/qwen2vl-grammar_correct_1fps_64maxframes_holistic_encoding'
    os.makedirs(jsonl_root, exist_ok=True)
    all_jsonls = find_all_files(raw_root, '.jsonl')
    output_jsonls = [os.path.join(jsonl_root, f'{index}.jsonl') for index in range(len(all_jsonls))]
    data = [(input_jsonl, output_jsonl) for input_jsonl, output_jsonl in zip(all_jsonls, output_jsonls)]
    data_cur_rank = dist_split_files(data)
    for input_jsonl, output_jsonl in tqdm(data_cur_rank):
        data = read_jsonl_file(input_jsonl)
        results = multi_process_with_extend(pack_single_row, data, num_workers=512)
        results = filterout_repeat_images_for_mmq(results)
        dump_list_to_jsonl_file(output_jsonl, results)
    print('Done!')
        
    
