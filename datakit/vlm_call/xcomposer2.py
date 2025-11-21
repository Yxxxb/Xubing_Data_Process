import os
from typing import Tuple

import torch
from transformers import AutoModel, AutoTokenizer


def xcomposer2_7b() -> Tuple[AutoModel, AutoTokenizer]:
    """Instantiate the X-Composer-2-7B model and tokenizer.

    Returns:
        Tuple[AutoModel, AutoTokenizer]: The X-Composer-2-7B model
            and tokenizer.
    """
    torch.set_grad_enabled(False)
    LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))
    MODEL_PATH = os.environ.get(
        'XComposer2_MODEL_PATH',
        '/mnt/cephfs/bensenliu/wfs/weights/mm/opensource/internlm-xcomposer2-vl-7b'  # noqa
    )
    assert os.path.exists(
        MODEL_PATH), f'Model path {MODEL_PATH} does not exist'
    model = AutoModel.from_pretrained(MODEL_PATH,
                                      trust_remote_code=True,
                                      device_map=f'cuda:{LOCAL_RANK}').eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH,
                                              trust_remote_code=True)
    return model, tokenizer
