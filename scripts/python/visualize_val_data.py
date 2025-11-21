import argparse
import os

import gradio as gr
import pandas as pd
from tqdm import tqdm

from datakit.utils.image import decode_base64_image_to_pil

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder',
                    type=str,
                    required=True,
                    help='path to data folder')
parser.add_argument('--val-dataset-names',
                    type=str,
                    nargs='+',
                    required=True,
                    help='names of validation datasets')

parser.add_argument('--show-num',
                    type=int,
                    default=1,
                    help='number of columns to show')


class MMDataViewer:
    """Visualize the validation data.

    Args:
        show_num (int): number of data to show. Defaults to 3.
        port (int): port to launch the gradio app. Defaults to 8081.
    """
    def __init__(self, show_num: int = 3, port: int = 8081) -> None:
        self.show_num = show_num
        self.port = port

    def _updata_gr(self, data):
        self.gr_data = data

    def _map_gr(self, item):
        question = item['question']
        image = decode_base64_image_to_pil(item['image'])
        text = f'image size:{image.size}\n\n' + question
        return image, text

    def _gr_display_image_select_file(self, dataname, index):
        return self._map_gr(self.gr_data[dataname].loc[index])

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
    runner = MMDataViewer(show_num=args.show_num)
    root = args.data_folder
    selected_val_datasets = args.val_dataset_names
    validation_sets = [
        os.path.join(root, dataset) for dataset in selected_val_datasets
    ]
    data = {
        os.path.basename(validation_set): pd.read_csv(validation_set, sep='\t')
        for validation_set in tqdm(validation_sets,
                                   desc='Registering data ...')
    }
    runner.visualize(data)
