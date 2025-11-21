import json
import subprocess
import time
from typing import Optional

import requests


def call_mock_search(query: str,
                     temperature: float = 0.6,
                     top_p: float = 0.95,
                     max_tokens: int = 2048,
                     seed: int = 42,
                     retry: int = 0) -> Optional[str]:
    """Call the mock search server to get the response of
        the input query.

    Args:
        query (str): The input query.
        temperature (float, optional): The temperature of the
            softmax function. Defaults to 0.6.
        top_p (float, optional): The top-p probability
            threshold. Defaults to 0.95.
        max_tokens (int, optional): The maximum number of
            tokens to generate. Defaults to 2048.
        seed (int, optional): The seed for generating
            the response. Defaults to 42.
        retry (int, optional): The number of retries. Defaults to 0.

    Returns:
        Optional[str]: The response of the input query.
    """
    msgs = [{'role': 'user', 'content': query}]
    url = 'http://mmsprllmappsvr.production.polaris:24892/llm_gateway/call_server/xubingye-Qwen3-32A3B-Instruct'  # noqa
    if retry >= 3:
        return None
    try:
        req = {
            'model': './model',
            'messages': msgs,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'seed': seed,
            'stream': False,
            'stop': [' <end_of_turn>'],
        }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, data=json.dumps(req))
        json_dict = json.loads(response.text)
        result = json_dict['choices'][0]['message']['content']
        return result
    except Exception as e:
        print(f'Error: {e}')
        time.sleep(5)
        return call_mock_search(query, temperature, top_p, max_tokens, seed,
                                retry + 1)


def remote_server_utilization() -> None:
    """To maximize GPU utilization of remote server.
    """
    command = ['python', '/mnt/cephfs/bensenliu/exp_runs/server_loop.py']
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    pid = process.pid
    return pid
