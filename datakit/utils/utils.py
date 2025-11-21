import json
import subprocess
from datetime import datetime
from typing import List

import requests
from torch import Tensor, nn


def get_device(p: nn.Module | Tensor):
    """Get the device of a module or tensor."""
    if isinstance(p, Tensor):
        return p.device
    return next(p.parameters()).device


def get_dtype(p: nn.Module | Tensor):
    """Get the dtype of a module or tensor."""
    if isinstance(p, Tensor):
        return p.dtype
    return next(p.parameters()).dtype


def concat_conversations_to_string(conversations: List[dict]) -> str:
    """Concatenate a list of conversations to a string.

    Args:
        conversations (List[dict]): A list of conversations.
            Each conversation is a dictionary with keys "role" and "text".

    Returns:
        str: The concatenated string.
    """
    conversation_str = ''
    for conversation in conversations:
        conversation_str += f'{conversation["role"]}: {conversation["text"]}\n'
    return conversation_str.strip()


def time_to_seconds(time_str: str) -> float:
    """Convert a time string to seconds.

    Args:
        time_str (str): A time string in the format of "%H:%M:%S.%f".

    Returns:
        float: The total seconds.
    """
    time_obj = datetime.strptime(time_str, '%H:%M:%S.%f')
    total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1e6  # noqa
    return total_seconds


def run_command_and_get_return_code(command: str) -> tuple:
    """Run a command and get the return code, stdout,
        and stderr.

    Args:
        command (str): The command to run.

    Returns:
        tuple: A tuple containing the return code, stdout,
            and stderr.
    """
    result = subprocess.run(command, capture_output=True, text=True)
    return_code = result.returncode
    stdout = result.stdout
    stderr = result.stderr

    return return_code, stdout, stderr


def send_messages_to_bot(webhook_url: str,
                         message_type: str = 'text',
                         chat_id: str = 'bensenliu',
                         message: str = 'Hello, world!') -> int:
    """Send messages to a bot via a webhook.

    Args:
        webhook_url (str): The webhook URL of the bot.
        message_type (str, optional): The message type.
            Defaults to 'text'."
        chat_id (str, optional): The chat ID of the bot.
            Defaults to 'bensenliu'.
        message (str, optional): The message to send.
            Defaults to 'Hello, world!'.

    Returns:
        int: The status code of the response.
    """
    headers = {'Content-Type': 'application/json'}
    data = {
        'chatid': chat_id,
        'msgtype': message_type,
        message_type: {
            'content': message
        }
    }
    response = requests.post(webhook_url,
                             headers=headers,
                             data=json.dumps(data))
    return response.status_code
