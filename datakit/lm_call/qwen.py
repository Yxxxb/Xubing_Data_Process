import os
from typing import Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer


def qwen_1_5_7b(
        auto: bool = True) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Initialize Qwen-v1.5-7b model and tokenizer.

    Args:
        auto (bool, optional): The device to use. Defaults to True.

    Returns:
        Tuple[AutoTokenizer, AutoModelForCausalLM]: Tuple of
            tokenizer and model.
    """
    MODEL_PATH = os.environ.get(
        'QWEN_1_5_7B_MODEL_PATH',
        '/mnt/cephfs/bensenliu/wfs/weights/mm/opensource/Qwen1.5-7B-Chat'
    )  # noqa
    assert os.path.exists(
        MODEL_PATH), f'Model path {MODEL_PATH} does not exist'
    LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    device = 'auto' if auto else f'cuda:{LOCAL_RANK}'

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,
                                                 torch_dtype='auto',
                                                 device_map=device,
                                                 trust_remote_code=True)

    return tokenizer, model


def qwen_1_5_inference(prompt: str,
                       tokenizer: AutoTokenizer,
                       model: AutoModelForCausalLM,
                       max_new_tokens: int = 256) -> str:
    """Inference with Qwen-v1.5 model.

    Args:
        prompt (str): Input prompt.

    Returns:
        str: Output text.
    """
    LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))
    messages = [{
        'role': 'system',
        'content': 'You are a helpful assistant.'
    }, {
        'role': 'user',
        'content': prompt
    }]
    text = tokenizer.apply_chat_template(messages,
                                         tokenize=False,
                                         add_generation_prompt=True)
    model_inputs = tokenizer([text],
                             return_tensors='pt').to(f'cuda:{LOCAL_RANK}')
    generated_ids = model.generate(model_inputs.input_ids,
                                   max_new_tokens=max_new_tokens)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(
            model_inputs.input_ids, generated_ids)  # noqa
    ]
    response = tokenizer.batch_decode(generated_ids,
                                      skip_special_tokens=True)[0]

    return response
