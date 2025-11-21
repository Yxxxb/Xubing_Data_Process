import functools
import os
from datetime import datetime
from pprint import pprint
from typing import Callable, List

import numpy as np
import pytz
from tqdm import tqdm
from transformers.image_transforms import resize
from transformers.image_utils import ChannelDimension
from transformers.models.qwen2_vl import Qwen2VLImageProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from .distributed import (barrier_all_processes, dist_split_files,
                          get_distributed_env, gpu_utilization, kill_process)
from .files import (dump_list_to_jsonl_file, filterout_repeat_images_for_mmq,
                    find_all_files, read_jsonl_file)
from .image import (check_image_integrity, decode_base64_image_to_np,
                    decode_base64_image_to_pil, encode_np_to_base64_image)
from .mp import multi_process_with_append, multi_process_with_extend

MAX_PIXELS = 14 * 14 * 4 * 160
image_processor = Qwen2VLImageProcessor.from_pretrained(
    '/mnt/cephfs/bensenliu/wfs/weights/mm/opensource/Qwen2-VL-7B-Instruct')


def add_dummy_image(data: list, data_file: str) -> List[dict]:
    dummy_image = {
        'id': data_file,
        'type': 'image',
        'content': 'This is the content of dummy image'
    }
    data.append(dummy_image)
    return data


def pack_single_row_image(item: dict,
                          fps: int = 2,
                          maxframes: int = 64,
                          maxpixels: int = MAX_PIXELS) -> List[dict]:
    """
    Pack a single row of images into a conversation template.
    Args:
        item (dict): A dictionary containing the base64 images and
                    conversations.
        fps (int): Frames per second.
        maxframes (int): Maximum number of frames.
        maxpixels (int): Maximum number of pixels.
    Returns:
        A list of conversation templates.
    """
    base64_image_dict = item[
        'base64_images'] if 'base64_images' in item else item['base64_image']
    conversations = item['conversations']
    keys = list(base64_image_dict.keys())
    image_id = item['id']
    if len(keys) == 0 or base64_image_dict[keys[0]] is None:
        return None
    image_templates = []
    image_text = ''
    seq_lens = dict()

    is_video = 'fps' in list(item.keys())
    if 'total_frames' not in list(item.keys()):
        item['total_frames'] = len(base64_image_dict)

    if is_video:
        ori_fps = item['fps']
        fps = min(fps, ori_fps)
        if item['total_frames'] / ori_fps > maxframes / fps:
            fps = ori_fps * ((maxframes + 1) / (item['total_frames'] + 1))
        current_second = 0
        frame_count = 0

        for image_name, base64_image in base64_image_dict.items():
            utc_now = datetime.now(pytz.utc)
            image_name = image_name + utc_now.strftime(
                '%Y-%m-%d %H:%M:%S.%f %Z%z') + str(
                    np.random.randint(1000000)) + str(
                        np.random.randint(1000000)) + str(
                            np.random.randint(1000000))
            current_second += 1 / ori_fps
            if current_second >= 1 / fps:
                if not check_image_integrity(base64_image):
                    return None
                try:
                    np_image = decode_base64_image_to_np(base64_image)
                    height, width = np_image.shape[:2]
                    resized_height, resized_width = smart_resize(
                        height, width, max_pixels=maxpixels)

                    image = resize(np_image,
                                   size=(resized_height, resized_width),
                                   resample=3,
                                   input_data_format=ChannelDimension.LAST)

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
                current_second -= 1 / fps
                frame_count += 1
                if frame_count > maxframes:
                    break
    else:
        total_frames = len(base64_image_dict)
        if total_frames > maxframes:
            base64_image_dict = {
                k: base64_image_dict[k]
                for k in list(base64_image_dict.keys())[::total_frames //
                                                        maxframes][:maxframes]
            }
        # interleave and few shot data
        image_text_list = []
        for image_name, base64_image in base64_image_dict.items():
            utc_now = datetime.now(pytz.utc)
            image_name = image_name + utc_now.strftime(
                '%Y-%m-%d %H:%M:%S.%f %Z%z') + str(
                    np.random.randint(1000000)) + str(
                        np.random.randint(1000000)) + str(
                            np.random.randint(1000000))
            if not check_image_integrity(base64_image):
                return None
            try:
                np_image = decode_base64_image_to_np(base64_image)
                height, width = np_image.shape[:2]
                resized_height, resized_width = smart_resize(
                    height, width, max_pixels=maxpixels)

                image = resize(np_image,
                               size=(resized_height, resized_width),
                               resample=3,
                               input_data_format=ChannelDimension.LAST)

                if image.shape[-1] == 1:
                    image = image.repeat(3, axis=-1)
                base64_image = encode_np_to_base64_image(image)
                seq_len = int(resized_width / 28 * resized_height / 28)
                seq_lens[image_name] = seq_len
            except Exception as e:
                print(f'Error in {image_name}: {e}')
                return []
            image_text += f'<img>{image_name}</img>'
            image_text_list.append(f'<img>{image_name}</img>')
            image_template = {
                'id': image_name,
                'type': 'image',
                'content': base64_image
            }
            image_templates.append(image_template)
    for i in range(len(image_text_list)):
        conversations[i]['text'] = image_text_list[i] + conversations[i]['text']
    # conversations[0]['text'] = image_text + conversations[0]['text']
    conversation_template = {
        'type': 'conversation',
        'seq_lens': seq_lens,
        'id': image_id,
        'conversations': conversations
    }
    return [conversation_template] + image_templates


def pack_single_row_holistic(item: dict,
                             fps: int = 2,
                             maxframes: int = 64,
                             maxpixels: int = MAX_PIXELS) -> List[dict]:
    """
       Pack single row of images into a conversation template in holistic
       manner.
       Args:
           item: dict, the item to be packed
           fps: int, the fps of the video
           maxframes: int, the maximum number of frames to be packed
           maxpixels: int, the maximum number of pixels of the image
        Returns:
        A list of conversation templates.
    """
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
    ori_fps = item['fps']
    fps = min(fps, ori_fps)
    if item['total_frames'] / ori_fps > maxframes / fps:
        fps = ori_fps * (maxframes / item['total_frames'])
    current_second = 0
    frame_count = 0
    try:
        for base64_name, base64_image in base64_image_dict.items():
            current_second += 1 / ori_fps
            if current_second >= 1 / fps:
                if not check_image_integrity(base64_image):
                    return None
                pil_images.append(decode_base64_image_to_pil(base64_image))
                utc_now = datetime.now(pytz.utc)
                base64_name = base64_name + utc_now.strftime(
                    '%Y-%m-%d %H:%M:%S.%f %Z%z') + str(
                        np.random.randint(1000000)) + str(
                            np.random.randint(1000000)) + str(
                                np.random.randint(1000000))
                image_text += f'<img>{base64_name}</img>'
                image_template = {
                    'id': base64_name,
                    'type': 'image',
                    'content': base64_image
                }
                image_templates.append(image_template)
                current_second -= 1 / fps
                frame_count += 1
                if frame_count > maxframes:
                    break
        output = image_processor(images=None,
                                 videos=pil_images,
                                 max_pixels=maxpixels)
    except Exception as e:
        print(e)
        return None
    video_grid_thw = output['video_grid_thw']
    seq_len = int(np.prod(video_grid_thw)) // 4
    conversations[0]['text'] = image_text + conversations[0]['text']
    conversation_template = {
        'type': 'conversation',
        'seq_lens': {
            image_id: seq_len
        },
        'max_pixels': maxpixels,
        'is_video': True,
        'id': image_id,
        'conversations': conversations
    }
    return [conversation_template] + image_templates


def process_jsonl_file(jsonl,
                       pack_function: Callable[[dict, int], List[dict]] = None,
                       output_root: str = None) -> str:
    sub_folder_name = os.path.basename(os.path.dirname(jsonl))
    jsonl_name = os.path.basename(jsonl)
    if os.path.basename(output_root) != sub_folder_name:
        output_file = os.path.join(output_root, sub_folder_name, jsonl_name)
    else:
        output_file = os.path.join(output_root, jsonl_name)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    try:
        data = read_jsonl_file(jsonl)
        output_list = []
        for i in tqdm(range(len(data))):
            output_list.extend(pack_function(data[i]))
            output_list = filterout_repeat_images_for_mmq(output_list)
        output_list = add_dummy_image(output_list, jsonl)
        dump_list_to_jsonl_file(output_file, output_list)
    except Exception as e:
        print(f'Error in {jsonl}: {e}')
    print(f'file {jsonl} processed successfully')
    return jsonl


def raw_to_jsonl(root_list: str,
                 is_holistic_list: str = 'false',
                 num_workers: int = 64,
                 fps: int = 2,
                 maxframes: int = 64,
                 alert_path: str = '/mnt/cephfs/haichengwang/envs/loop_test',
                 parallel_line: bool = False,
                 rename: str = None,
                 maxpixels: int = MAX_PIXELS) -> List[str]:
    """
    Convert raw jsonl files to jsonl files, applicable for all
    image/video datasets

    example usage:
    1. python convert_raw_to_jsonl_qwen2.py --root-list
              /mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category-video/QA/finevideo/data
              --num-workers 256 --is-holistic-list false

    2. python convert_raw_to_jsonl_qwen2.py --root-list
              /mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category-video/QA/finevideo/data,
              /mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/CHART/ChartQA
              --num-workers 256
              --is-holistic-list true,false

    Args:
        root_list (str): string of root directories of raw jsonl files,
                         separated by commas
        is_holistic_list (str): string of boolean values for whether the data
                                is holistic. Separated by commas.
                                Defaults to 'false'
        num_workers (int): number of workers for multiprocessing.
                           Defaults to 64.
        fps (int): frames per second. Defaults to 2
        maxframes (int): maximum number of frames. Defaults to 64
        alert_path (str): path to alert file.
                          Defaults to '/mnt/cephfs/haichengwang/envs/loop_test'
        parallel_line (bool): whether to process each line in parallel or
                              each file in parallel.Defaults to False
        rename (str): rename the output directory. Defaults to None
        maxpixels (int): maximum number of pixels. Defaults to MAX_PIXELS

    Returns:
        output_root_list: list of output directories
    """
    gpu_utilization()
    if alert_path is None:
        alert_path = os.path.dirname(__file__)
    root_list = root_list.split(',')
    is_holistic_list = is_holistic_list.split(',')
    if len(is_holistic_list) == 1:
        is_holistic_list = [is_holistic_list[0] for _ in range(len(root_list))]
    assert len(root_list) == len(
        is_holistic_list
    ), 'The number of is_holistic_list must be the same of root_list'

    output_root_list = []
    world_size, rank, _ = get_distributed_env()
    for i, root in tqdm(enumerate(root_list)):
        root = root.strip()
        is_holistic = is_holistic_list[i].strip()
        is_holistic = True if is_holistic == 'true' else False
        if not os.path.exists(root):
            print(f'{root} does not exist')
            continue

        output_root = root.replace('/data/', '/jsonl/')
        rename = f'fps{fps}_maxframe{maxframes}' if rename is None else rename
        output_root = os.path.join(output_root, rename)
        if not os.path.exists(output_root):
            os.makedirs(output_root, exist_ok=True)
        output_root_list.append(output_root)
        with open(os.path.join(output_root, 'config.txt'), 'w') as f:
            f.write(f'root: {root}\n')
            f.write(f'output_root: {output_root}\n')
            f.write(f'is_holistic: {is_holistic}\n')
            f.write(f'num_workers: {num_workers}\n')
            f.write(f'fps: {fps}\n')
            f.write(f'maxframes: {maxframes}\n')
            f.write(f'maxpixels: {maxpixels//(14*14*4)}\n')

        all_jsonls = find_all_files(root, 'jsonl')
        jsonls_cur_rank = dist_split_files(all_jsonls)
        pprint(f"""Current rank {os.environ['RANK']}
                processing {len(jsonls_cur_rank)} files""")
        success_files = []
        failed_files = []
        if is_holistic:
            pack_single_row = pack_single_row_holistic
        else:
            pack_single_row = pack_single_row_image
        pack_single_row = functools.partial(pack_single_row,
                                            fps=fps,
                                            maxframes=maxframes,
                                            maxpixels=maxpixels)
        if parallel_line:
            for jsonl in tqdm(jsonls_cur_rank):
                sub_folder_name = os.path.basename(os.path.dirname(jsonl))
                jsonl_name = os.path.basename(jsonl)
                if os.path.basename(root) != sub_folder_name:
                    output_file = os.path.join(output_root, sub_folder_name,
                                               jsonl_name)
                else:
                    output_file = os.path.join(output_root, jsonl_name)
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                try:
                    data = read_jsonl_file(jsonl)
                    data = multi_process_with_extend(pack_single_row,
                                                     data,
                                                     num_workers=num_workers)
                    data = filterout_repeat_images_for_mmq(data)
                    data = add_dummy_image(data, jsonl)
                    dump_list_to_jsonl_file(output_file, data)
                    success_files.append(jsonl)
                except Exception as e:
                    print(f'Error in {jsonl}: {e}')
                    failed_files.append(jsonl)
            if len(failed_files) > 0:
                print(f'Failed files in {root}: {failed_files}')
            else:
                print(f'All files in {root} processed successfully')
        else:
            # process_jsonl_file("/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_cot/geoqa+/data/grammar_correct_vlm_infer_Qwen72B_complexity_vllm_pe/03.jsonl", output_root="/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_cot/geoqa+/jsonl/grammar_correct_vlm_infer_Qwen72B_complexity_vllm_pe/qwen_2_5_vl_72b_vllm_cot_test3")
            # process_jsonl_file("/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_few_shot/geoqa+/data/grammar_correct/00.jsonl", output_root="/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_complex_results_few_shot/geoqa+/jsonl/grammar_correct/qwen_2_5_vl_72b_vllm_few_shot")
            process_jsonl_file1 = functools.partial(
                process_jsonl_file,
                pack_function=pack_single_row,
                output_root=output_root)
            _ = multi_process_with_append(process_jsonl_file1,
                                          jsonls_cur_rank,
                                          num_workers=num_workers)

    print(f'All tasks in rank {rank} processed successfully')

    barrier_all_processes(task_name='jsonl', root=alert_path)
    kill_process()  # kill max GPU utilization process
    return output_root_list
