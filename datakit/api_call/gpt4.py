import hashlib
import hmac
import time
from typing import Optional

import requests

MAX_RETRY = 5


def calcAuthorization(source, appkey):
    timestamp = int(time.time())
    signStr = 'x-timestamp: %s\nx-source: %s' % (timestamp, source)
    sign = hmac.new(appkey.encode('utf8'), signStr.encode('utf8'),
                    hashlib.sha256).digest()
    return sign.hex(), timestamp


def get_prediction(url, req, header):
    resp = requests.post(url, json=req, headers=header).json()
    response = resp['response']
    return response


def query_gpt4(prompt: str,
               model_version: str = 'azure-gpt-4-turbo') -> Optional[str]:
    """Query GPT-4 model to generate a response to a prompt.

    Args:
        prompt (str): The prompt to generate a response for.
        model_version (str, optional): The GPT-4 model version to use.
            Defaults to 'azure-gpt-4-turbo'.

    Returns:
        Optional[str]: The generated response, or None if the request failed.
    """
    appid = 'appgxe4pptof9sc9t91'
    appkey = 'PyyWrmnXSqyfkVDqRtTdmavvYxyQIIjr'

    keyword = 'odyssey'
    source = f'{keyword}-gpt-4'

    auth, timestamp = calcAuthorization(source, appkey)

    header = {
        'X-AppID': appid,
        'X-Source': source,
        'X-Timestamp': f'{timestamp}',
        'X-Authorization': auth,
    }
    url = 'http://ichat.woa.com/api/chat_completions'
    req = {
        'model': model_version,
        'messages': [{
            'role': 'user',
            'content': prompt
        }]
    }
    for i in range(MAX_RETRY):
        try:
            return get_prediction(url, req, header)
        except Exception as e:  # noqa
            print(f'第{i+1}次请求失败，原因：{e}')
            time.sleep(2**i)
    print('请求失败')
    return None
