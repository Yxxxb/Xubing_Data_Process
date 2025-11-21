import re
import zlib


def string_repetition_detection(ori_string: str,
                                global_ratio: float = 4,
                                local_ratio: float = 3.5,
                                window_size: int = 1000) -> bool:
    """
    Detect if a string is a repetition of the original string.

    Args:
        ori_string (str): The original string.
        global_ratio (float): The ratio of the compressed string length to the
            original string length. Defaults to 4.
        local_ratio (float): The ratio of the compressed string length to the
            original string length in each sliding window. Defaults to 3.5.
        window_size (int): The size of the sliding window.
            Defaults to 1000.
    Returns:
        bool: True if the string is a repetition of the original string,
            False otherwise.
    """
    compressed = zlib.compress(ori_string.encode('utf-8'))
    compressed_length = len(compressed)
    ori_length = len(ori_string.encode('utf-8'))
    if ori_length / (1e-8 + compressed_length) > global_ratio:
        return True
    else:
        for i in range(len(ori_string) // window_size):
            window_block = ori_string[i * window_size:(i + 1) * window_size]
            window_bytes = window_block.encode('utf-8')
            window_compressed = zlib.compress(window_bytes)
            window_ratio = len(window_bytes) / (1e-8 + len(window_compressed))
            if window_ratio > local_ratio:
                return True
    return False


def filter_string_for_fox(string: str) -> str:
    """Filter a string for Fox evaluation.

    Args:
        string (str): The string to be filtered.

    Returns:
        str: The filtered string.
    """
    prompt = (
        'Please extract all the text from the image with the following requirements:\n'  # noqa
        '1. Return tables in HTML format.\n'
        '2. Return all other text in Markdown format.')
    string = string.replace('---', '')
    string = string.replace('*', '')
    string = string.replace('<sup>', '')
    string = string.replace('</sup>', '')
    string = string.replace('\n', '')
    string = string.replace('>', '')
    string = string.replace('$', '')
    string = string.replace('^', '')
    string = string.replace('{', '')
    string = string.replace('}', '')
    string = string.replace('#', '')
    string = string.replace(prompt, '')
    pattern = r'\[Image\]\(.*?\)'
    string = re.sub(pattern, '', string)
    return string
