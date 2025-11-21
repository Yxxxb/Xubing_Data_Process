from typing import List

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


class Qwen2VLLMWrapper:
    """vllm wrapper for Qwen2VL.

    Args:
        model_name_or_path (str): path to the model or model name
        temperature (float, optional): temperature for sampling.
            Defaults to 0.1.
        top_p (float, optional): top-p for sampling. Defaults to 0.001.
        repetition_penalty (float, optional): repetition penalty for sampling.
            Defaults to 1.05.
        tensor_parallel_size (int, optional): tensor parallel size for
            sampling. Defaults to 1.
        max_tokens (int, optional): max tokens for sampling. Defaults to 256.
    """
    def __init__(self,
                 model_name_or_path: str,
                 temperature: float = 0.1,
                 top_p: float = 0.001,
                 repetition_penalty: float = 1.05,
                 tensor_parallel_size: int = 1,
                 max_tokens: int = 256) -> None:
        self.llm = LLM(
            model=model_name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            # fix currently
            limit_mm_per_prompt={
                'image': 10,
                'video': 10
            })
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)

    def generate(self, inputs: List[dict]) -> str:
        """Generate responses for the given prompts.

        Args:
            inputs (List[dict]): a list of inputs, each prompt is a
                dictionary with keys 'image', 'video', 'text'.
        """
        messages = [{
            'role': 'system',
            'content': 'You are a helpful assistant.'
        }]
        user_messages = []
        for input_ in inputs:
            dtype = input_['type']
            if dtype == 'image':
                message = {
                    'type': dtype,
                    dtype: input_['content'],
                    'max_pixels': 12845056
                }
            else:
                message = {'type': dtype, dtype: input_['content']}
            user_messages.append(message)
        messages.append({'role': 'user', 'content': user_messages})
        prompt = self.processor.apply_chat_template(messages,
                                                    tokenize=False,
                                                    add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        if video_inputs is not None:
            mm_data['video'] = video_inputs
        llm_inputs = {
            'prompt': prompt,
            'multi_modal_data': mm_data,
        }
        outputs = self.llm.generate([llm_inputs],
                                    sampling_params=self.sampling_params,
                                    use_tqdm=False)
        response = outputs[0].outputs[0].text
        return response
