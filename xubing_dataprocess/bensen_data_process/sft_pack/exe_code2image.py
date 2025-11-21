from datakit.utils.files import read_jsonl_file, dump_list_to_jsonl_file
from datakit.utils.image import encode_pil_to_base64_image
from PIL import Image
from datakit.utils.mp import multi_process_with_append
import os
import random


PROMPT = [
    "请提供能够生成此图的 Python 代码。",
    "请分享用于绘制该图的 Python 代码。",
    "请给出生成该图像的 Python 代码。",
    "请提供绘制这张图的 Python 代码。",
    "请给出可以生成这幅图的 Python 代码。",
    "请提供用于创建此图的 Python 代码。",
    "请分享可以绘制这张图片的 Python 代码。",
    "请给出能够绘制该图的 Python 代码。",
    "请提供生成这张图片的 Python 代码。",
    "请分享用于生成这幅图的 Python 代码。",
    "Please provide the Python code that can generate this graph.",
    "Please share the Python code used to plot this graph.",
    "Please give the Python code to generate this image.",
    "Please provide the Python code to draw this graph.",
    "Please give the Python code that can generate this figure.",
    "Please provide the Python code used to create this graph.",
    "Please share the Python code that can draw this picture.",
    "Please give the Python code that can plot this graph.",
    "Please provide the Python code to generate this picture.",
    "Please share the Python code used to generate this figure."
]


def exe_code2image(item):
    sample_id = item['id']
    code = item['code']
    show_code = 'plt.show()'
    save_code = 'plt.savefig("/tmp/{}.png")'.format(sample_id)
    code = code.replace(show_code, save_code)
    try:
        exec(code)
        img = Image.open('/tmp/{}.png'.format(sample_id))
        base64_image = encode_pil_to_base64_image(img)
        os.remove('/tmp/{}.png'.format(sample_id))
        code = code.replace(save_code, show_code)
    except Exception as e:
        print(e)
        return None
    prompt = random.choice(PROMPT)
    template = {
        'id': sample_id,
        'base64_image': {
            sample_id: base64_image
        }, 
        'conversations': [
            {
                'role': 'user',
                'text': prompt
            },
            {
                'role': 'assistant',
                'text': code
            }
        ]
    }
    return template


if __name__ == '__main__':
    input_file = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/CODE/image2code/raw/image2code.jsonl'
    output_file = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category/CODE/image2code/data/grammar_correct/image2code.jsonl'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    data = read_jsonl_file(input_file)
    results = multi_process_with_append(exe_code2image, data, num_workers=128)
    dump_list_to_jsonl_file(output_file, results)