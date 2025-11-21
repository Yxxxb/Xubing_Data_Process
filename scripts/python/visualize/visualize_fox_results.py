import argparse
import json
import os

import gradio as gr
from PIL import Image

from datakit.utils.files import find_all_files
from datakit.utils.string import filter_string_for_fox

parser = argparse.ArgumentParser()
parser.add_argument('--input-folder', type=str, required=True)

folder_mapping = {
    'cn':
    '/mnt/cephfs/bensenliu/wfs/mm_datasets/eval/focus_benchmark_test/cn_pdf_png',  # noqa
    'en':
    '/mnt/cephfs/bensenliu/wfs/mm_datasets/eval/focus_benchmark_test/en_pdf_png'  # noqa
}


def visualize_fox_results(json_file, index):
    with open(json_file, 'r') as f:
        data = json.load(f)
    sample = data[index]
    target = filter_string_for_fox(sample['label'])
    prediction = filter_string_for_fox(sample['answer'])
    print(sample['answer'])
    image_name = sample['image']
    image_path = os.path.join(folder_mapping[image_name.split('_')[0]],
                              image_name)
    image = Image.open(image_path)
    return image, target, prediction


def main(args):
    all_jsons = find_all_files(args.input_folder, 'json')
    inputs = []
    outputs = []
    with gr.Column():
        with gr.Row():
            inputs.extend([
                gr.Dropdown(choices=all_jsons, label='Select a JSON file'),
                gr.Slider(0, 112, step=1)
            ])
        with gr.Row():
            outputs.append(gr.Image(type='pil', elem_id='image-container'))
            with gr.Column():
                outputs.append(gr.Textbox(label='Target'))
                outputs.append(gr.Textbox(label='Prediction'))
    return gr.Interface(
        visualize_fox_results,
        inputs,
        outputs,
        title='Fox Results Visualization',
        description='Visualize the results of Fox benchmark.').launch(
            server_name='0.0.0.0', server_port=8081)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
