import argparse
import os

import gradio as gr
from PIL import Image

from datakit import Qwen2VLLMWrapper

parser = argparse.ArgumentParser()
parser.add_argument('--model-path',
                    type=str,
                    default='.',
                    help='The path to the model file')
parser.add_argument('--max-pixels',
                    type=int,
                    default=1024 * 1024,
                    help='The maximum number of pixels in the input image')
parser.add_argument('--tensor-parallel-size',
                    type=int,
                    default=4,
                    help='The size of tensor parallelism')

args = parser.parse_args()

# set the environment variable for the model path
model_path = args.model_path
model = Qwen2VLLMWrapper(model_path,
                         max_tokens=1024,
                         tensor_parallel_size=args.tensor_parallel_size)


def answer_question(image: Image.Image, question: str) -> str:
    """Answer a question based on an image.

    Args:
        image (Image.Image): The input image.
        question (str): The question to answer.

    Returns:
        str: The answer to the question.
    """
    height, width = image.size
    print(f'Received image with size {image.size} and question: {question}')
    if height * width > args.max_pixels:
        # resize the image to fit the maximum number of pixels,
        # and maintain the aspect ratio
        ratios = (height * width) / args.max_pixels
        height = int(height / ratios**0.5)
        width = int(width / ratios**0.5)
        image = image.resize((height, width))
        print(f'Resized image to {image.size}')
    tmp_image_path = 'tmp.jpg'
    image.save(tmp_image_path)
    messages = [{
        'type': 'image',
        'content': './tmp.jpg'
    }, {
        'type': 'text',
        'content': question
    }]
    response = model.generate(messages)
    os.remove(tmp_image_path)
    return response


demo = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.Image(type='pil', label='Upload an Image'),
        gr.Textbox(lines=2,
                   placeholder='Enter your question here...',
                   label='Question')
    ],
    outputs='text',
    title='Qwen2-VL Demo',
    description=('Upload an image and ask a question about it. '
                 'The model will try to answer your question based '
                 'on the image content.'))

if __name__ == '__main__':

    demo.launch(server_name='0.0.0.0', server_port=8081)
