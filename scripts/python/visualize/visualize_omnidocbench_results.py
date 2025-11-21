import argparse
import os

import gradio as gr
from PIL import Image

from datakit.utils.files import find_all_files

parser = argparse.ArgumentParser()
parser.add_argument('--input-folder', type=str, required=True)

IMAGE_FOLDER = '/mnt/cephfs/bensenliu/wfs/mm_datasets/eval/OmniDocBench/images'
ALL_MDS = None


def visualize_fox_results(index):
    md_file = ALL_MDS[index]
    image_file = os.path.join(IMAGE_FOLDER, os.path.basename(md_file))
    image_file = image_file.replace('md', 'jpg')
    with open(md_file) as f:
        md_content = f.read()
    image = Image.open(image_file)
    return image, md_content


def main(args):
    global ALL_MDS
    all_mds = find_all_files(args.input_folder, 'md')
    ALL_MDS = all_mds
    inputs = []
    outputs = []
    with gr.Column():
        with gr.Row():
            inputs.extend(
                [gr.Slider(0, len(all_mds) - 1, label='Index', step=1)])
        with gr.Row():
            outputs.append(gr.Image(type='pil', elem_id='image-container'))
            with gr.Column():
                outputs.append(gr.Textbox(label='Prediction'))
    return gr.Interface(
        visualize_fox_results,
        inputs,
        outputs,
        title='OmniDocBench Results Visualization',
        description='Visualize the results of Fox benchmark.').launch(
            server_name='0.0.0.0', server_port=8081)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
