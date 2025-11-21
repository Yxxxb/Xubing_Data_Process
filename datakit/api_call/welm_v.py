from typing import List, Optional

import requests


def query_welm_v(prompt: str,
                 images: List[str],
                 addr: str = 'localhost:8081') -> Optional[str]:
    """Query WELM-V model for completion.

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
            f' <end_of_turn>user\n{image_prompt}{prompt} <end_of_turn>assistant\n',  # noqa
            'n_to_generate': 512,
            'images': images,
            'temperature': 0.2,
            'top_p': 0.8,
            'stop_tokens': [' <end_of_turn>'],
        },
        proxies={
            'http': None,
            'https': None
        })

    try:
        resp = resp.json()['text'][0].strip(' <end_of_turn>')
    except Exception as e:
        print(f'Error: {e}')
        return None

    return resp
