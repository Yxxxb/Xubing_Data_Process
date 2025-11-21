import os
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def internlm_7b(
        auto: bool = True) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Instantiate the InternLM-7b model and tokenizer.

    Args:
        auto (bool): Whether to use automatic device placement.
            Defaults to True.

    Returns:
        tokenizer: The tokenizer for the model.
        model: The model itself.
    """
    MODEL_PATH = os.environ.get(
        'INTERNLM_MODEL_PATH',
        '/mnt/cephfs/bensenliu/exp_runs/weights/nlp/internlm2-chat-7b')  # noqa
    assert os.path.exists(
        MODEL_PATH), f'Model path {MODEL_PATH} does not exist'
    LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH,
                                              trust_remote_code=True)

    device = 'auto' if auto else f'cuda:{LOCAL_RANK}'

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map=device,
                                                 trust_remote_code=True)
    model = model.eval()

    return tokenizer, model


def internlm_20b(
        auto: bool = True) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Instantiate the InternLM-20b model and tokenizer.

    Args:
        auto (bool): Whether to use automatic device placement.
            Defaults to True.

    Returns:
        tokenizer: The tokenizer for the model.
        model: The model itself.
    """
    MODEL_PATH = os.environ.get(
        'INTERNLM_MODEL_PATH',
        '/mnt/cephfs/bensenliu/exp_runs/weights/nlp/internlm2-chat-20b'
    )  # noqa
    assert os.path.exists(
        MODEL_PATH), f'Model path {MODEL_PATH} does not exist'
    LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH,
                                              trust_remote_code=True)

    device = 'auto' if auto else f'cuda:{LOCAL_RANK}'

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map=device,
                                                 trust_remote_code=True)
    model = model.eval()

    return tokenizer, model


def internlm_inference(model_name: str = '7b') -> None:
    """Instantiate the InternLM model and tokenizer for inference.

    Args:
        model_name: The name of the model to instantiate.
            Defaults to '7b'.
    """
    assert model_name in [
        '7b', '20b'
    ], f'Invalid model name {model_name}, must be either 7b or 20b'  # noqa
    if model_name == '7b':
        tokenizer, model = internlm_7b()
    else:
        tokenizer, model = internlm_20b()

    proceed = input(
        f'Do you want to proceed with the {model_name} model? (y/n) ')

    while True and proceed.lower() == 'y':
        prompt = input('Enter a prompt: ')
        output, _ = model.chat(tokenizer, prompt)
        print(output)
        proceed = input('Do you want to continue? (y/n) ')

    print('Exit')
