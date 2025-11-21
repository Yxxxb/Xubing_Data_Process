from typing import List, Optional

import requests


def query_llama3_v(prompt: str,
                   images: List[str],
                   addr: str = 'localhost:8081') -> Optional[str]:
    """Query LLaMA3-V model for completion.

    Args:
        prompt (str): The prompt to complete.
        images (List[str]): A list of base64 encoded images.
        addr (str, optional): The address of the WELM-V server.
            Defaults to "localhost:8081".

    Returns:
        Optional[str]: The completion text.
    """
    image_prompt = ''
    for i, _ in enumerate(images):
        image_prompt += f'<img_{i}></img_{i}>' + '\n'
    resp = requests.post(
        f'http://{addr}/completion',
        json={
            'prompt':
            f'<|start_header_id|>user<|end_header_id|>\n\n{image_prompt}{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n',  # noqa
            'n_to_generate': 512,
            'images': images,
            'temperature': 0.2,
            'top_p': 0.8,
            'stop_tokens': ['<|eot_id|>'],
        },
        proxies={
            'http': None,
            'https': None
        })

    try:
        resp = resp.json()['text'][0].strip('<|eot_id|>')
    except Exception as e:
        print(f'Error: {e}')
        return None

    return resp
