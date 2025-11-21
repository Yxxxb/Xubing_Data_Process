import json
from typing import List

import requests


def query_qwen2vl(image_paths: List[str],
                  prompt: str,
                  max_tokens: int = 1024,
                  repetition_penalty: float = 1.05) -> str:
    """Query Qwen2VL model to generate text for a given prompt and images.

    Args:
        image_paths (List[str]): A list of image paths.
        prompt (str): A prompt string.
        max_tokens (int, optional): Maximum number of tokens to generate.
            Defaults to 1024.
        repetition_penalty (float, optional): Repetition penalty.
            Defaults to 1.05.

    Returns:
        str: The generated text.
    """
    url = 'http://localhost:8081/v1/chat/completions'
    image_content = [{
        'type': 'image_url',
        'image_url': {
            'url': image_path
        }
    } for image_path in image_paths]
    content = [{'type': 'text', 'text': prompt}]
    content.extend(image_content)
    data = {
        'model': 'Qwen/Qwen2-VL-7B-Instruct',
        'messages': [{
            'role': 'user',
            'content': content,
        }],
        'max_tokens': max_tokens,
        'repetition_penalty': repetition_penalty
    }
    response = requests.post(url, json=data)
    response = json.loads(response.text)
    response = response['choices'][0]['message']['content']
    return response
