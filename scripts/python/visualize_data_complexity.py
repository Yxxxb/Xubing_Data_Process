import argparse
import random
import re
from typing import List

import gradio as gr
from PIL import Image
from tqdm import tqdm

from datakit.utils.files import find_all_files, read_jsonl_file
from datakit.utils.image import decode_base64_image_to_pil, draw_bounding_box

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder',
                    type=str,
                    required=True,
                    help='path to data folder')
parser.add_argument(
    '--show-num',
    type=int,
    default=1,
)
parser.add_argument('--text-name', type=str, default='text')
parser.add_argument('--port', type=int, default=8081)
parser.add_argument('--visualize-pretrain', action='store_true')
parser.add_argument('--num-jsonls',
                    type=int,
                    default=10,
                    help='number of jsonls to visualize')

pattern = r'\[Image\]\(([-+]?\d*\.\d+|\d+),\s*([-+]?\d*\.\d+|\d+),\s*([-+]?\d*\.\d+|\d+),\s*([-+]?\d*\.\d+|\d+)\)'  # noqa
TEXT_NAME = 'text'


def concat_images_in_grid(image_list: List[Image.Image],
                          max_images_per_row: int = 20) -> Image.Image:
    """
    Concatenate images in a grid.
    """
    num_images = len(image_list)
    num_rows = (num_images + max_images_per_row - 1) // max_images_per_row

    max_width_per_image = max(image.width for image in image_list)
    max_height_per_image = max(image.height for image in image_list)

    total_width = max_width_per_image * min(max_images_per_row, num_images)
    total_height = max_height_per_image * num_rows

    new_image = Image.new('RGB', (total_width, total_height))

    x_offset = 0
    y_offset = 0

    for i, image in enumerate(image_list):
        new_image.paste(image, (x_offset, y_offset))
        x_offset += max_width_per_image

        if (i + 1) % max_images_per_row == 0:
            x_offset = 0
            y_offset += max_height_per_image

    return new_image


class MMDataViewer:
    """Visualize the pre-training and visual instruction tuning data.

    Args:
        visualize_pretrain (bool): whether to visualize pre-training data.
            Defaults to True.
        show_num (int): number of data to show. Defaults to 3.
        port (int): port to launch the gradio app. Defaults to 8081.
    """
    def __init__(self,
                 visualize_pretrain: bool = True,
                 show_num: int = 3,
                 port: int = 8081) -> None:
        self.visualize_pretrain = visualize_pretrain
        self.show_num = show_num
        self.port = port

    def _updata_gr(self, data):
        self.gr_data = data

    def _map_gr(self, item):
        if self.visualize_pretrain:
            caption = item['xcomposer2_caption']
            image = decode_base64_image_to_pil(item['base64_image'])
            text = f'image size:{image.size}\n\n' + caption
        else:
            conversations = item['conversations']
            conversations_str = ''
            for conversation in conversations:
                role = conversation['role']
                text = conversation[
                    TEXT_NAME] if role == 'assistant' else conversation['text']
                conversations_str += f'{role}: {text}\n'

            # complexity
            if 'difficult_conversations' in item:
                difficult_conversations = item['difficult_conversations']
                difficult_conversations_str = ''
                for conversation in difficult_conversations:
                    role = conversation['role']
                    text = conversation[
                        'qwen2.5vl-72b'] if role == 'assistant' else conversation['text']
                    difficult_conversations_str += f'{role}: {text}\n'
                conversations_str += f'\nDifficult Conversations:\n{difficult_conversations_str}'
            if 'easy_conversations' in item:
                easy_conversations = item['easy_conversations']
                easy_conversations_str = ''
                for conversation in easy_conversations:
                    role = conversation['role']
                    text = conversation[
                        'qwen2.5vl-3b'] if role == 'assistant' else conversation['text']
                    easy_conversations_str += f'{role}: {text}\n'
                conversations_str += f'\nEasy Conversations:\n{easy_conversations_str}'

            if 'base64_image' not in item:
                base64_images = None
            else:
                base64_images = []
                for image_key in item['base64_image']:
                    base64_images.append(item['base64_image'][image_key])
            if base64_images is None:
                image = None
                image_size = 'N/A'
            else:
                images = [
                    decode_base64_image_to_pil(image)
                    for image in base64_images
                ]
                image = concat_images_in_grid(images)
                matches = re.findall(pattern, conversations_str)
                if matches:
                    for match in matches:
                        bbox = [float(x) for x in match]
                        image = draw_bounding_box(image, bbox)
                image_size = image.size
            text = f'image size:{image_size}\n\n' + conversations_str
        return image, text

    def _gr_display_image_select_file(self, dataname, index):
        return self._map_gr(self.gr_data[dataname][index])

    def visualize(self, data: dict) -> None:
        """Visualize image-text(or conversation) pairs.

        Args:
            data (dict): a dict of data for each file.
        """

        self._updata_gr(data)
        custom_css = """
        #image-container img {
            height: auto;
            width: 500px;
        }
        """
        with gr.Blocks(css=custom_css) as demo:
            inputs, outputs = [], []
            with gr.Row():
                for i in range(self.show_num):
                    with gr.Column():
                        inputs.extend([
                            gr.Dropdown(choices=list(data.keys()),
                                        label=f'Select DataFrame {i}'),
                            gr.Slider(0,
                                      max(len(df) - 1 for df in data.values()),
                                      step=1,
                                      label=f'Index {i}')
                        ])
            with gr.Row():
                for i in range(self.show_num):
                    with gr.Column():
                        outputs.extend([
                            gr.Image(type='pil', elem_id='image-container'),
                            gr.Textbox(label=f'Description {i}')
                        ])
            for i in range(len(inputs) // 2):
                inputs[2 * i].change(
                    self._gr_display_image_select_file,
                    inputs=[inputs[2 * i], inputs[2 * i + 1]],
                    outputs=[outputs[2 * i], outputs[2 * i + 1]])
                inputs[2 * i + 1].change(
                    self._gr_display_image_select_file,
                    inputs=[inputs[2 * i], inputs[2 * i + 1]],
                    outputs=[outputs[2 * i], outputs[2 * i + 1]])
            demo.launch(server_name='0.0.0.0', server_port=self.port)


if __name__ == '__main__':
    args = parser.parse_args()
    TEXT_NAME = args.text_name
    runner = MMDataViewer(visualize_pretrain=args.visualize_pretrain,
                          show_num=args.show_num,
                          port=args.port)
    root = args.data_folder
    jsonls = find_all_files(root, '.jsonl')
    # random.shuffle(jsonls)
    jsonls = jsonls[:args.num_jsonls]

    jsonls = [
            # '/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B/ChartQA.jsonl', 
            '/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B/ChartQA_single_word.jsonl', 
            # '/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B/clevr_math_5w.jsonl', 
            '/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B/clevr_math_5w_single_word.jsonl', 
            '/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B/geo3k.jsonl', 
            '/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B/geoqa+.jsonl', 
            '/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B/iconqa_choose_txt.jsonl', 
            '/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B/iconqa_fill_blank.jsonl', 
            # '/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B/infovqa.jsonl', 
            # '/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B/infovqa_sample.jsonl', 
            # '/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B/infovqa_single_word.jsonl', 
            '/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B/infovqa_single_word_sample.jsonl', 
            # '/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B/mapqa.jsonl', 
            '/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B/mm_ai2d.jsonl', 
            '/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B/scienceqa.jsonl', 
            # '/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B/sharegpt4v_ref.jsonl', 
            # '/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B/sharegpt4v_ref_sample.jsonl', 
            # '/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B/super_clever.jsonl', 
            '/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B/super_clever_sample.jsonl', 
            '/mnt/cephfs/xubingye/wfs/datasets/sft-category/complexity_sum/Qwen_25_vl_3B_72B/tqa_sample.jsonl'
              ]

    data = {
        '_'.join(fn.split('/')[-4:]): read_jsonl_file(fn)
        for fn in tqdm(jsonls, desc='Registering data ...')
    }
    runner.visualize(data)
