import os
from typing import List, Tuple

from PIL import Image

from datakit.utils.image import (decode_base64_image_to_pil,
                                 encode_pil_to_base64_image)
from datakit.utils.large_image import factorize_number


def construct_mapping_dict(max_splits: int = 12) -> dict:
    """Construct a mapping dictionary for the given max_splits.

    Args:
        max_splits (int, optional): The maximum number of splits.
            Defaults to 12.

    Returns:
        dict: A mapping dictionary for the given max_splits.
    """
    mapping_dict = {}
    for i in range(1, max_splits + 1):
        factor_list = factorize_number(i)
        for factor in factor_list:
            ratio = factor[0] / factor[1]
            if ratio not in mapping_dict:
                mapping_dict[ratio] = [factor]
            else:
                mapping_dict[ratio].append(factor)
    return mapping_dict


def save_image_list(image_list: List[Image.Image], save_folder: str) -> None:
    """Save a list of images to a folder.

    Args:
        image_list (List[Image.Image]): A list of images.
        save_folder (str): The folder to save the images to.
    """
    os.makedirs(save_folder, exist_ok=True)
    for i, image in enumerate(image_list):
        image.save(os.path.join(save_folder, f'{i}.png'))


def resize_to_best_size(image: Image.Image, best_slices: tuple,
                        width_slices: int, height_slices: int,
                        sub_image_size: int) -> Image.Image:
    """Resize an image to the best size for the given number of slices.

    Args:
        image (Image.Image): The image to resize.
        best_slices (tuple): The best number of slices for the image.
        width_slices (int): The number of horizontal slices.
        height_slices (int): The number of vertical slices.
        sub_image_size (int): The size of the sub-images.

    Returns:
        Image.Image: The resized image.
    """
    width, height = image.size
    best_width_slices, best_height_slices = best_slices
    if width_slices < height_slices:
        new_image_width = best_width_slices * sub_image_size
        new_image_height = int(height / width * new_image_width)
    else:
        new_image_height = best_height_slices * sub_image_size
        new_image_width = int(width / height * new_image_height)
    new_image = image.resize((new_image_width, new_image_height), resample=2)
    return new_image


def compute_strides(height: int, width: int, sub_image_size: int,
                    slices: Tuple[int, int]) -> Tuple[int, int]:
    """Compute the strides for the given image size and slices.

    Args:
        height (int): The height of the image.
        width (int): The width of the image.
        sub_image_size (int): The size of the sub-images.
        slices (Tuple[int, int]): The number of horizontal and vertical slices.

    Returns:
        Tuple[int, int]: The strides for the given image size and slices.
    """
    slice_width, slice_height = slices
    if slice_width > 1:
        stride_x = (width - sub_image_size) // (slice_width - 1)
    else:
        stride_x = 0
    if slice_height > 1:
        stride_y = (height - sub_image_size) // (slice_height - 1)
    else:
        stride_y = 0
    return stride_x, stride_y


def sliding_window_crop(image: Image.Image, window_size: int,
                        slices: Tuple[int, int]) -> List[Image.Image]:
    """Crop an image into sub-images using a sliding window.

    Args:
        image (Image.Image): The image to crop.
        window_size (int): The size of the sub-images.
        slices (Tuple[int, int]): The number of horizontal and vertical slices.

    Returns:
        List[Image]: A list of cropped images.
    """
    width, height = image.size
    stride_x, stride_y = compute_strides(height, width, window_size, slices)
    sub_images = []
    if stride_x == 0:
        stride_x = window_size

    if stride_y == 0:
        stride_y = window_size
    for y in range(0, height - window_size + 1, stride_y):
        for x in range(0, width - window_size + 1, stride_x):
            sub_image = image.crop((x, y, x + window_size, y + window_size))
            sub_images.append(sub_image)
    return sub_images


def add_image_token(image_ids: List[str]) -> str:
    """Add image tokens to the image ids.

    Args:
        image_ids (List[str]): A list of image ids.

    Returns:
        str: The image text with image tokens.
    """
    text = ''
    for _, image_id in enumerate(image_ids):
        cur_image = f'<img>{image_id}</img>'
        text += cur_image
    return text


def find_best_slices(width_slices: int,
                     height_slices: int,
                     aspect_ratio: float,
                     max_splits: int = 12) -> list:
    """Find the best slices for the given image size and aspect ratio.

    Args:
        width_slices (int): The number of horizontal slices.
        height_slices (int): The number of vertical slices.
        aspect_ratio (float): The aspect ratio of the image.
        max_splits (int, optional): The maximum number of splits.
            Defaults to 12.

    Returns:
        list: the best slices for the given image.
    """
    mapping_dict = construct_mapping_dict(max_splits)
    if aspect_ratio < 1:
        mapping_dict = {
            k: v
            for k, v in mapping_dict.items() if k <= aspect_ratio
        }
    elif aspect_ratio > 1:
        mapping_dict = {
            k: v
            for k, v in mapping_dict.items() if k >= aspect_ratio
        }
    # find the value which key is the closest to the ratio
    best_ratio = min(mapping_dict.keys(), key=lambda x: abs(x - aspect_ratio))
    # best_image_sizes is a list of image sizes
    best_image_sizes = mapping_dict[best_ratio]
    # find the image_size whose area is closest to the current image size
    best_slices = min(
        best_image_sizes,
        key=lambda x: abs(x[0] * x[1] - width_slices * height_slices))
    return best_slices


def construct_overlap_image_template(base64_image: str,
                                     image_id: str,
                                     image_size: int = 336,
                                     max_crop_slices: int = 12,
                                     debug: bool = False,
                                     save_folder: str = None,
                                     add_thumbnail: bool = False,
                                     return_split_images: bool = False,
                                     do_resize: bool = False,
                                     **kwargs) -> tuple:
    """Construct image templates for a given image.

    Args:
        base64_image (str): The base64 encoded image.
        image_id (str): The id of the image.
        image_size (int, optional): The size of the image.
            Defaults to 336.
        max_crop_slices (int, optional): The maximum number of slices.
            Defaults to 12.
        debug (bool, optional): Whether to save the sub-images.
            Defaults to False.
        save_folder (str, optional): The folder to save the sub-images.
            Defaults to None.
        add_thumbnail (bool, optional): Whether to add a thumbnail.
            Defaults to False.
        return_split_images (bool, optional): Whether to return the split
            images. Defaults to False.
        do_resize (bool, optional): Whether to resize the image to fit the
            maximum number of slices. Defaults to False.

    Returns:
        tuple: A tuple containing the image templates and the image text.
    """
    pil_image = decode_base64_image_to_pil(base64_image)
    width, height = pil_image.size
    ratio = width / height
    if ratio > max_crop_slices or ratio < 1 / max_crop_slices:
        if do_resize:
            print(
                f'Resizing image to fit maximum number of slices ({max_crop_slices})'  # noqa
            )  # noqa
            if width > height:
                new_width = max_crop_slices * height
                new_height = height
            else:
                new_width = width
                new_height = max_crop_slices * width
            pil_image = pil_image.resize((new_width, new_height), resample=2)
            width, height = pil_image.size
            ratio = width / height
        else:
            print(
                f'Image aspect ratio ({ratio:.2f}) is out of range: ({1/max_crop_slices:.2f}, {max_crop_slices:.2f})'  # noqa
            )
            return None, None
    width_slices = width / image_size
    height_slices = height / image_size
    best_slices = find_best_slices(width_slices, height_slices, ratio,
                                   max_crop_slices)
    pil_image = resize_to_best_size(pil_image, best_slices, width_slices,
                                    height_slices, image_size)
    width, height = pil_image.size
    sub_images = sliding_window_crop(pil_image, image_size, best_slices)
    if add_thumbnail:
        thumbnail_image = pil_image.resize((image_size, image_size),
                                           resample=2)
        sub_images.append(thumbnail_image)
    base64_sub_images = [
        encode_pil_to_base64_image(sub_image) for sub_image in sub_images
    ]
    if debug:
        assert save_folder is not None
        save_image_list(sub_images, save_folder)
    image_templates = []
    image_ids = []
    for i, _ in enumerate(sub_images):
        image_template = {
            'id': f'{image_id}_{i}',
            'type': 'image',
            'content': base64_sub_images[i]
        }
        image_templates.append(image_template)
        image_ids.append(f'{image_id}_{i}')
    image_text = add_image_token(image_ids)
    if return_split_images:
        return image_templates, image_text, sub_images
    return image_templates, image_text
