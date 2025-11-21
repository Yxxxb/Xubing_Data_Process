import time
from typing import Optional

import requests

MAX_RETRY = 5


def query_chatgpt(prompt: str) -> Optional[str]:
    """Query the GPT3.5 model to generate a response to the given prompt.

    Args:
        prompt (str): The prompt to generate a response for.

    Returns:
        Optional[str]: The generated response, or None if the request failed.
    """
    url = 'http://mmsprastraapi.production.polaris:8080/astra/models/opensource_ichat'  # noqa
    data = {
        'model': 'gpt-3.5-turbo',
        'messages': [
            {
                'role': 'user',
                'content': prompt
            },
        ],
    }
    for i in range(MAX_RETRY):
        try:
            resp = requests.post(url,
                                 json=data,
                                 proxies={
                                     'http': None,
                                     'https': None
                                 })
            resp = resp.json()
            response = resp['response']
            return response
        except Exception as e:  # noqa
            time.sleep(2**i)
    print('请求失败')
    return None
