import json
from typing import Optional

import requests

API_KEY = 'LBjb4dDGq0fx1XZlE1LGoJBi'
SECRET_KEY = 'rjjLtpiCq2jyuyrfjMn0DUbcV6xF7Tmk'


def query_ernie(prompt: str) -> Optional[str]:
    """Query ERNIE chatbot API to get the response of the input prompt.

    Args:
        prompt (str): The input prompt to be sent to ERNIE chatbot.

    Returns:
        Optional[str]: The response of the input prompt from ERNIE chatbot.
    """
    url = 'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=' + get_access_token(  # noqa
    )

    payload = json.dumps({
        'messages': [{
            'role': 'user',
            'content': prompt
        }],
        'disable_search': False,
        'enable_citation': False
    })
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.request('POST', url, headers=headers, data=payload)
        response = json.loads(response.text)
        return response['result']
    except Exception as e:
        print(e)
        return None


def get_access_token():
    url = 'https://aip.baidubce.com/oauth/2.0/token'
    params = {
        'grant_type': 'client_credentials',
        'client_id': API_KEY,
        'client_secret': SECRET_KEY
    }
    return str(requests.post(url, params=params).json().get('access_token'))
