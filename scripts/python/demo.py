import json
import os
import re
import tempfile
from typing import List

import gradio as gr
import requests
from PIL import Image
from sglang.utils import wait_for_server

from datakit.utils.image import decode_base64_image_to_pil, draw_bounding_box
from datakit.utils.video import extract_frames_decord

pattern = r'\[Image\]\(([-+]?\d*\.\d+|\d+),\s*([-+]?\d*\.\d+|\d+),\s*([-+]?\d*\.\d+|\d+),\s*([-+]?\d*\.\d+|\d+)\)'  # noqa


def convert_messages_to_sglang_messages(messages: List[dict]) -> List[dict]:
    sglang_messages = []
    for message in messages:
        if message['type'] == 'text':
            sglang_messages.append({'type': 'text', 'text': message['text']})
        else:
            sglang_messages.append({
                'type': 'image_url',
                'image_url': {
                    'url': message['image']
                }
            })
    return sglang_messages


def query_sglang(messages: List[dict],
                 temperature: float = 0.0,
                 max_new_tokens: int = 2048) -> str:
    data = {
        'model': 'WePoints',
        'messages': [{
            'role': 'user',
            'content': messages
        }],
        'max_new_tokens': max_new_tokens,
        'temperature': temperature,
        'do_sample': False,
    }
    response = requests.post('http://127.0.0.1:8082/v1/chat/completions',
                             json=data)
    response = json.loads(response.text)
    response = response['choices'][0]['message']['content']
    print(response)
    return response


def answer_question(file: tempfile.NamedTemporaryFile,
                    question: str,
                    max_new_tokens: int = 2048,
                    temperature: float = 0.0) -> str:
    """Answer a question based on an image.

    Args:
        file (tempfile.NamedTemporaryFile): The input file.
        question (str): The question to answer.
        max_new_tokens (int): maximum number of new tokens to generate.
            Defaults to 2048.
        temperature (float): temperature for sampling.
            Defaults to 0.0.

    Returns:
        str: The answer to the question.
    """
    max_new_tokens = int(max_new_tokens)
    temperature = float(temperature)
    print(f'Received question: {question}')
    if 'jpg' in file.name or 'png' in file.name or 'jpeg' in file.name:
        image = Image.open(file.name).convert('RGB')
        images = [image]
    elif 'avi' in file.name or 'mp4' in file.name or 'mov' in file.name:
        images = extract_frames_decord(file.name)
        images = [decode_base64_image_to_pil(image) for image in images]
    else:
        raise ValueError('Unsupported file type')
    # save images and create image content
    content = []
    for i, image in enumerate(images):
        image_path = os.path.join(tempfile.gettempdir(), f'image_{i}.jpg')
        image.save(image_path)
        content.append(dict(type='image', image=image_path))
    content.append(dict(type='text', text=question))
    content = convert_messages_to_sglang_messages(content)
    response = query_sglang(content, temperature, max_new_tokens)
    # remove temporary images
    for i, image in enumerate(images):
        os.remove(os.path.join(tempfile.gettempdir(), f'image_{i}.jpg'))
    matches = re.findall(pattern, response)
    if matches:
        for match in matches:
            bbox = [float(x) for x in match]
            images[0] = draw_bounding_box(images[0], bbox)
    return response, images[0]


demo = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.inputs.File(label='Upload Video or Image', type='file'),
        gr.Textbox(lines=2,
                   placeholder='Enter your question here...',
                   label='Question'),
        gr.inputs.Number(default=2048, label='Maximum New Tokens'),
        gr.inputs.Slider(minimum=0.0,
                         maximum=1.0,
                         step=0.01,
                         default=0.0,
                         label='Temperature')
    ],
    outputs=[
        gr.Textbox(label='Answer'),
        gr.Image(type='pil', label='Annotated Image')
    ],
    title='WePOINTS',
    description=('Upload an image and ask a question about it. '
                 'The model will try to answer your question based '
                 'on the image content.'))

if __name__ == '__main__':
    wait_for_server('http://127.0.0.1:8082')
    demo.launch(server_name='0.0.0.0', server_port=8081)
