import base64
import hashlib
import time
from typing import Optional

from openai import OpenAI

MAX_RETRIES = 5


def get_sha1(text):
    if isinstance(text, str):
        text = text.encode()

    s1 = hashlib.sha1()
    s1.update(text)

    return s1.hexdigest()


def make_sign_header(appid, secret):
    timestamp = str(int(time.time()))
    text = '{0}{1}{2}'.format(appid, secret, timestamp)

    sign = get_sha1(text=text)
    base_token = ','.join([appid, timestamp, sign])
    header = base64.b64encode(base_token.encode()).decode()
    return header


def query_search_gpt4(query: str,
                      model_version: str = 'azure-gpt-4-turbo',
                      retry: int = 0) -> Optional[str]:
    """Query GPT-4 model to generate a response to the input query.

    Args:
        query (str): input query.
        model_version (str, optional): GPT-4 model version.
            Defaults to "azure-gpt-4-turbo".
        retry (int, optional): retry count. Defaults to 0.

    Returns:
        Optional[str]: generated response.
    """
    sign = make_sign_header('letian', '41cf55cdc1ced591e5243419e231354c')
    client = OpenAI(
        api_key=sign,
        base_url='http://mmsearchopenaiproxy.polaris:8080/v1',
    )
    try:
        chat_completion = client.chat.completions.create(messages=[{
            'role':
            'user',
            'content':
            query,
        }],
                                                         model=model_version)
    except Exception as e:
        if retry >= MAX_RETRIES:
            print('error, max retries reached')
            return None
        print(f'failed to query gpt4: {e}, retrying {retry+1}/{MAX_RETRIES}')
        time.sleep(2**retry)
        return query_search_gpt4(query, model_version, retry + 1)

    return chat_completion.choices[0].message.content
