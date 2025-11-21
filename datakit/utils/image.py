import base64
import io
from typing import List

import numpy as np
from PIL import Image, ImageDraw


def save_base64_image(base64_str, save_path):
    """Save a base64 string as an image file.

    Args:
        base64_str (str): The base64 string of the image.
        save_path (str): The path to save the image.
    """
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data)).convert('RGB')
    img.save(save_path)


def check_image_integrity(base64_image: str) -> bool:
    """Check if the base64 string is a valid image.

    Args:
        base64_image (str): The base64 string of the image.


    Returns:
        bool: True if the image is valid, False otherwise.
    """
    try:
        image = Image.open(io.BytesIO(
            base64.b64decode(base64_image))).convert('RGB')
        _ = image.size
        return True
    except Exception:
        return False


def decode_base64_image_to_np(base64_image: str) -> np.ndarray:
    """Decode a base64 string of an image to a numpy array.

    Args:
        base64_image (str): The base64 string of the image.

    Returns:
        np.ndarray: The numpy array of the image.
    """
    img_data = base64.b64decode(base64_image)
    img = Image.open(io.BytesIO(img_data))
    return np.array(img)


def decode_base64_image_to_pil(base64_image: str) -> Image:
    """Decode a base64 string of an image to a PIL Image.

    Args:
        base64_image (str): The base64 string of the image.

    Returns:
        Image: The PIL Image of the image.
    """
    img_data = base64.b64decode(base64_image)
    img = Image.open(io.BytesIO(img_data)).convert('RGB')
    return img


def encode_np_to_base64_image(np_image: np.ndarray) -> str:
    """Encode a numpy array of an image to a base64 string.

    Args:
        np_image (np.ndarray): The numpy array of the image.

    Returns:
        str: The base64 string of the image.
    """
    img = Image.fromarray(np_image)
    img = img.convert('RGB')
    buffered = io.BytesIO()
    img.save(buffered, format='JPEG')
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str


def encode_pil_to_base64_image(pil_image: Image) -> str:
    """Encode a PIL Image of an image to a base64 string.

    Args:
        pil_image (Image): The PIL Image of the image.

    Returns:
        str: The base64 string of the image.
    """
    pil_image = pil_image.convert('RGB')
    buffered = io.BytesIO()
    pil_image.save(buffered, format='JPEG')
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str


def encode_bytes_to_base64_image(bytes_image: bytes) -> str:
    """Encode a bytes of an image to a base64 string.

    Args:
        bytes_image (bytes): The bytes of the image.

    Returns:
        str: The base64 string of the image.
    """

    base64_image = base64.b64encode(bytes_image).decode('utf-8')
    return base64_image


def encode_numpy_array_to_base64(np_array: np.ndarray) -> str:
    """Encode a numpy array of an image to a base64 string.

    Args:
        np_array (np.ndarray): The numpy array of the image.

    Returns:
        str: The base64 string of the image.
    """
    np_array_bytes = np_array.tobytes()
    np_array_base64 = base64.b64encode(np_array_bytes).decode('utf-8')
    return np_array_base64


def center_crop_into_square_pil_image(image: Image.Image,
                                      center_crop_size: int = None
                                      ) -> Image.Image:
    """Center crop a PIL image, to make it square. Keep most information.

    Args:
        image (Image.Image): The PIL image to be cropped.
        new_shape (Tuple[int]): The new shape of the cropped image,
                Default to be the minimum of the original width and height.
    Returns:
        Image.Image: The cropped PIL image.
    """
    width, height = image.size
    crop_size = min(width, height)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = (width + crop_size) // 2
    bottom = (height + crop_size) // 2
    center_cropped_image = image.crop((left, top, right, bottom))

    if center_crop_size is None:
        return center_cropped_image
    resized_image = center_cropped_image.resize(
        (center_crop_size, center_crop_size), Image.Resampling.BICUBIC)
    return resized_image


def rotate_images_in_item(item: dict, degrees: List[int]) -> List[dict]:
    """Rotate images in an item.

    Args:
        item (dict): The base64 image in item to be rotated.
        degrees (List[int]): The degrees to rotate the images.

    Returns:
        List[dict]: The rotated images in the item.
    """
    item_list = [item]
    for degree in degrees:
        cur_base64_image_dict = {}
        for _, base64_image in item['base64_image'].items():
            pil_image = decode_base64_image_to_pil(base64_image)
            pil_image = pil_image.rotate(degree, expand=True)
            base64_image = encode_pil_to_base64_image(pil_image)
            cur_base64_image_dict[str(hash(base64_image))] = base64_image
        cur_item = item.copy()
        cur_item['base64_image'] = cur_base64_image_dict
        item_list.append(cur_item)
    return item_list


def resize_image_to_max_size(image: Image.Image,
                             max_pixels: int) -> Image.Image:
    """Resize an image to a maximum number of pixels.

    Args:
        image (Image.Image): The PIL image to be resized.
        max_pixels (int): The maximum number of pixels of
            the resized image.

    Returns:
        Image.Image: The resized PIL image.
    """
    height, width = image.size
    if height * width <= max_pixels:
        return image
    ratios = (height * width) / max_pixels
    height = int(height / ratios**0.5)
    width = int(width / ratios**0.5)
    image = image.resize((height, width))
    return image


def draw_bounding_box(image: Image.Image, bbox: List[float]) -> Image.Image:
    """Draw a bounding box on an image.

    Args:
        image (Image.Image): The input image.
        bbox (List[float]): The bounding box coordinates in the
            format of [x_min, y_min, x_max, y_max].

    Returns:
        Image.Image: The image with the bounding box drawn.
    """
    draw = ImageDraw.Draw(image)
    width, height = image.size

    x_min = int(bbox[0] * width)
    y_min = int(bbox[1] * height)
    x_max = int(bbox[2] * width)
    y_max = int(bbox[3] * height)

    draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=3)
    return image


def draw_bounding_boxes(image: Image.Image, bbox: List[List[float]]) -> Image.Image:
    """Draw a bounding box on an image.

    Args:
        image (Image.Image): The input image.
        bbox (List[float]): The bounding box coordinates in the
            format of [x_min, y_min, x_max, y_max].

    Returns:
        Image.Image: The image with the bounding box drawn.
    """
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for box in bbox:
        x_min = int(box[0] * width)
        y_min = int(box[1] * height)
        x_max = int(box[2] * width)
        y_max = int(box[3] * height)

        draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=3)

    return image

    
def draw_bounding_boxes_with_labels(image: Image.Image, bbox: List[List[float]], labels: List[str]) -> Image.Image:
    """Draw bounding boxes with labels on an image.

    Args:
        image (Image.Image): The input image.
        bbox (List[List[float]]): The bounding box coordinates in the
            format of [x_min, y_min, x_max, y_max].
        labels (List[str]): The labels for each bounding box.

    Returns:
        Image.Image: The image with the bounding boxes and labels drawn.
    """
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for box, label in zip(bbox, labels):
        x_min = int(box[0] * width)
        y_min = int(box[1] * height)
        x_max = int(box[2] * width)
        y_max = int(box[3] * height)

        draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=3)
        draw.text((x_min, y_min), label, fill='red')

    return image
