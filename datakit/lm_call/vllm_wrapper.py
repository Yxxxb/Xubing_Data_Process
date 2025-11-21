try:
    from vllm import LLM, SamplingParams
except ImportError:
    print(
        'vLLM is not installed. Please install it by running `pip install vllm`.'  # noqa
    )
from transformers import AutoTokenizer


class vLLMWrapper:
    """Wrapper for HuggingFace model vLLM inference.

    Args:
        model_name_or_path (str): The path or name of the model.
        tensor_parallel_size (int): The number of GPUs to use for tensor
            parallelism. Defaults to 1.
        temperature (float): The temperature of the softmax. Defaults to 0.7.
        top_p (float): The top-p value for nucleus sampling. Defaults to 0.8.
        max_tokens (int): The maximum number of tokens to generate.
            Defaults to 1024.
        model_name (str): The name of the model. Defaults to 'qwen2.5'.
        repetition_penalty (float): The penalty for repetition.
            Defaults to 1.05.
        trust_remote_code (bool): Whether to trust the remote code of the
            model. Defaults to True.
    """
    def __init__(self,
                 model_name_or_path: str,
                 tensor_parallel_size: int = 1,
                 temperature: float = 0.7,
                 top_p: float = 0.8,
                 max_tokens: int = 1024,
                 model_name: str = 'qwen2.5',
                 repetition_penalty: float = 1.05,
                 trust_remote_code: bool = True) -> None:
        self.llm = LLM(model_name_or_path,
                       tensor_parallel_size=tensor_parallel_size,
                       trust_remote_code=trust_remote_code)
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty)  # noqa
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code)  # noqa
        self.model_name = model_name

    def apply_chat_template(self, prompt: str) -> str:
        """Apply chat template to the prompt.

        Args:
            prompt (str): The prompt to apply chat template.

        Returns:
            str: The prompt with chat template applied.
        """
        messages = []
        if self.model_name == 'qwen2.5':
            sys_message = {
                'role':
                'system',
                'content':
                'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'  # noqa
            }
            messages.append(sys_message)

        messages.append({'role': 'user', 'content': prompt})
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt

    def generate(self, prompt: str, use_tqdm: bool = True) -> str:
        """Generate answer using vLLM.

        Args:
            prompt (str): The prompt to generate answer.
            use_tqdm (bool): Whether to use tqdm to show progress bar.
                Defaults to True.

        Returns:
            str: The generated answer.
        """
        prompt = self.apply_chat_template(prompt)
        resp = self.llm.generate(prompt,
                                 sampling_params=self.sampling_params,
                                 use_tqdm=use_tqdm)
        resp = resp[0].outputs[0].text
        return resp
