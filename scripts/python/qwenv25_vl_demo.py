import argparse
import os
import re
import tempfile

import gradio as gr
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from datakit.utils.image import decode_base64_image_to_pil, draw_bounding_box
from datakit.utils.video import extract_frames_decord

parser = argparse.ArgumentParser()
parser.add_argument('--model-path',
                    type=str,
                    default='.',
                    help='The path to the model file')
args = parser.parse_args()

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path,
                                                           torch_dtype='auto',
                                                           device_map='auto')
processor = AutoProcessor.from_pretrained(args.model_path)
pattern = r'\[\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\]'  # noqa


def answer_question(file: tempfile.NamedTemporaryFile, question: str) -> str:
    """Answer a question based on an image.

    Args:
        file (tempfile.NamedTemporaryFile): The input file.
        question (str): The question to answer.

    Returns:
        str: The answer to the question.
    """
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
        content.append(dict(type='image', image='file://' + image_path))
    content.append(dict(type='text', text=question))
    messages = [{'role': 'user', 'content': content}]
    text = processor.apply_chat_template(messages,
                                         tokenize=False,
                                         add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text],
                       images=image_inputs,
                       videos=video_inputs,
                       padding=True,
                       return_tensors='pt')
    generated_ids = model.generate(**inputs, max_new_tokens=4096)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(generated_ids_trimmed,
                                         skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)
    response = output_text[0]
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
                   label='Question')
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

    demo.launch(server_name='0.0.0.0', server_port=8081)
