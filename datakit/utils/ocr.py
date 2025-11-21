from typing import List


def get_axis_min(items: list) -> List[dict]:
    """Get the top-left point of the axis.

    Args:
        items (list): A list of items

    Returns:
        List[dict]: A list of dictionaries containing the top-left point
            of the axis and the corresponding text.
    """
    results = []
    for item in items[0]:
        x_min, y_min = item[0][0]
        transcription = item[1][0]
        results.append({'point': [x_min, y_min], 'text': transcription})
    return results


def sort_key(item: dict) -> tuple:
    """Sort the items by their y-coordinate and x-coordinate.

    Args:
        item (dict): A dictionary containing the top-left point of the axis
            and the corresponding text.

    Returns:
        tuple: A tuple containing the y-coordinate and
            x-coordinate of the item.
    """
    return (item['point'][1], item['point'][0])


def get_structured_data(item: list) -> List[str]:
    """Get the structured data from the items.

    Args:
        item (list): A list of dictionaries containing the top-left point of
            the axis and the corresponding text and the corresponding text.

    Returns:
        List[str]: A list of strings containing the structured data.
    """
    structured_output = []
    current_line = []
    cur_y = item[0]['point'][1]

    for single_item in item:
        if abs(single_item['point'][1] - cur_y) < 10:
            current_line.append(single_item['text'])
        else:
            structured_output.append(' '.join(current_line))
            current_line = [single_item['text']]
            cur_y = single_item['point'][1]
    structured_output.append(' '.join(current_line))
    return structured_output


def reformat_ocr_results(results: list) -> str:
    """Reformat the OCR results.

    Args:
        results (list): A list of items from paddleocr.

    Returns:
        str: A string containing the structured data.
    """
    results = get_axis_min(results)
    results = sorted(results, key=sort_key)
    results = get_structured_data(results)
    results = '\n'.join(results)
    return results
