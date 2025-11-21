import json
import random
import subprocess
import time
from typing import List, Optional

import requests


def call_mock_search(query: str,
                     ip_list: List[str],
                     temperature: float = 0.6,
                     top_p: float = 0.95,
                     max_tokens: int = 2048,
                     seed: int = 42,
                     retry: int = 0) -> Optional[str]:
    """Call the mock search server to get the response of
        the input query.

    Args:
        query (str): The input query.
        ip_list (List[str]): The list of IP addresses of the
            mock search servers.
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
    urls = [f'http://{ip}:8000/v1/chat/completions' for ip in ip_list]
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
        url = random.choice(urls)
        response = requests.post(url, headers=headers, data=json.dumps(req))
        json_dict = json.loads(response.text)
        result = json_dict['choices'][0]['message']['content']
        return result
    except Exception as e:
        print(f'Error: {e}')
        time.sleep(5)
        return call_mock_search(query, ip_list, temperature, top_p, max_tokens,
                                seed, retry + 1)


def remote_server_utilization() -> None:
    """To maximize GPU utilization of remote server.
    """
    command = ['python', '/mnt/cephfs/bensenliu/exp_runs/server_loop.py']
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    pid = process.pid
    return pid


def call_qwen_25_vl(query: str,
                     image_path: str,
                     ip_list: List[str],
                     temperature: float = 0.6,
                     top_p: float = 0.95,
                     max_tokens: int = 2048,
                     seed: int = 42,
                     retry: int = 0) -> Optional[str]:
    """Call the qwen2.5-vl-72B server to get the response of
        the input query.

    Args:
        query (str): The input query.
        image_path (str): The path of the image.
        ip_list (List[str]): The list of IP addresses of the
            mock search servers.
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
    msgs = [
        {
            'role': 'system',
            'content': "You are a helpful assistant."
        },
        {
            'role': 'user', 
            'content': [
                {"type": "text", "text": query},
                {
                    "type": "image_url",
                    "image_url": {"url": f"file://{image_path}"},
                },
            ]
        }
    ]
    urls = [f'http://{ip}:8000/v1/chat/completions' for ip in ip_list]
    if retry >= 3:
        return None
    try:
        req = {
            'model': 'qwen2.5-vl',
            'messages': msgs,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'seed': seed,
            'stream': False,
            'stop': [' <end_of_turn>'],
        }
        headers = {'Content-Type': 'application/json'}
        url = random.choice(urls)
        response = requests.post(url, headers=headers, data=json.dumps(req))
        json_dict = json.loads(response.text)
        result = json_dict['choices'][0]['message']['content']
        return result
    except Exception as e:
        print(f'Error: {e}')
        time.sleep(5)
        return call_mock_search(query, ip_list, temperature, top_p, max_tokens,
                                seed, retry + 1)

if __name__ == '__main__':
    query = '这幅图描述了什么？'
    ip_list = ['29.181.53.104']
    image_path = '/mnt/cephfs/xubingye/tsp/ft_local/0811_edge/heatmap_edge_center_0.53869_edge_-0.02233.png'
    print(call_qwen_25_vl(query, image_path, ip_list))
