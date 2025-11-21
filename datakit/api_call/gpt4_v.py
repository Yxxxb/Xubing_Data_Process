import base64
from typing import List, Optional
from urllib.parse import urlparse

import requests

from .gpt4 import calcAuthorization

appid = 'appav8emxwlpbxpzk0h'
appkey = 'WsAdTMgPAhEeArIkhSSYiEtGJCWshnIM'
source = 'bensenliu'


def download_image(url: str) -> Optional[bytes]:
    """Download an image from a URL.

    Args:
        url (str): The URL of the image to download.

    Returns:
        Optional[bytes]: The downloaded image data, or None
            if an error occurred.
    """
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        return None


def encode_image(image_path: str) -> Optional[str]:
    """Encode an image file to base64 format.

    Args:
        image_path (str): The path of the image file to encode.

    Returns:
        Optional[str]: The encoded image data in base64 format, or None
            if an error occurred.
    """
    try:
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(e)
        return None


def encode_image_data_to_base64(image_data: bytes) -> Optional[str]:
    """Encode an image data to base64 format.

    Args:
        image_data (bytes): The image data to encode.

    Returns:
        Optional[str]: The encoded image data in base64 format, or None
            if an error occurred.
    """
    if image_data is not None:
        encoded_image = base64.b64encode(image_data)
        return encoded_image.decode('utf-8')
    else:
        return None


def is_url_or_local_path(path):
    parsed = urlparse(path)
    return parsed.scheme != '' and parsed.netloc != ''


def query_gpt4v(prompt: str,
                image_address_list: List[str],
                system_prompt=None,
                temperature=0.7) -> Optional[str]:
    """Query GPT-4-Vision API to generate a response to a prompt with images.

    Args:
        prompt (str): The prompt to generate a response for.
        image_address_list (List[str]): A list of image addresses, which
            can be either URLs or local file paths.
        system_prompt (str, optional): A system prompt to add to the beginning
            of the conversation. Defaults to None.
        temperature (float, optional): The temperature of the generated
            response. Defaults to 0.7.


    Returns:
        Optional[str]: The generated response, or None if an error occurred.
    """

    auth, timestamp = calcAuthorization(source, appkey)
    url = 'http://ichat.woa.com/api/chat_completions'
    header = {
        'X-AppID': appid,
        'X-Source': source,
        'X-Timestamp': f'{timestamp}',
        'X-Authorization': auth,
    }

    messages = []

    if system_prompt is not None:
        messages.append({'role': 'system', 'content': system_prompt})

    prompts = [{'type': 'text', 'text': prompt}]

    image_prompts = []
    for image_address in image_address_list:
        if is_url_or_local_path(image_address):
            image_data = download_image(image_address)
            if image_data is not None:
                encoded_image = encode_image_data_to_base64(image_data)
                if encoded_image is not None:
                    image_prompts.append({
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/jpeg;base64,{encoded_image}'
                        },
                    })
        else:
            encoded_image = encode_image(image_address)
            if encoded_image is not None:
                image_prompts.append({
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:image/jpeg;base64,{encoded_image}'
                    },
                })

    prompts.extend(image_prompts)

    messages.append({'role': 'user', 'content': prompts})
    req = {
        'model': 'azure-gpt-4-vision-preview',
        'messages': messages,
        'temperature': temperature,
    }
    try:
        resp = requests.post(url, json=req, headers=header)
        resp = resp.json()['response']
    except Exception as e:
        print(e)
        return None
    return resp
