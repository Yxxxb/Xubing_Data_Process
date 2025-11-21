import base64
import json
import pickle
import zipfile
from typing import Tuple


def read_all_files_in_zip(
        zip_file_path: str,
        extension: str = 'any') -> Tuple[zipfile.ZipFile, list]:  # noqa
    """Read all files in a zip file.

    Args:
        zip_file_path (str): Path to the zip file.
        extension (str, optional): File extension to filter. Defaults to 'any'.

    Returns:
        zipfile.ZipFile, list: Zip file object and list of file names.
    """
    zip_ref = zipfile.ZipFile(zip_file_path, 'r')
    file_list = [
        file_name for file_name in zip_ref.namelist()
        if file_name.endswith(extension) or extension == 'any'
    ]
    return zip_ref, file_list


def zip_read_json_file(zipfile_handler: zipfile.ZipFile,
                       file_name: str) -> dict:
    """Read a JSON file from a zip file.

    Args:
        zipfile_handler (zipfile.ZipFile): Zip file object.
        file_name (str): Name of the JSON file.

    Returns:
        dict: JSON data.
    """
    data = zipfile_handler.open(file_name).read()
    data = json.loads(data)
    return data


def zip_read_jsonl_file(zipfile_handler: zipfile.ZipFile,
                        file_name: str) -> list:
    """Read a JSONL file from a zip file.

    Args:
        zipfile_handler (zipfile.ZipFile): Zip file object.
        file_name (str): Name of the JSONL file.

    Returns:
        list: List of JSON data.
    """
    data = zipfile_handler.open(file_name).readlines()
    data = [json.loads(line) for line in data if line.strip()]
    return data


def zip_read_txt_file(zipfile_handler: zipfile.ZipFile, file_name: str) -> str:
    """Read a text file from a zip file.

    Args:
        zipfile_handler (zipfile.ZipFile): Zip file object.
        file_name (str): Name of the text file.

    Returns:
        str: Text data.
    """
    data = zipfile_handler.open(file_name).read()
    data = data.decode('utf-8')
    return data


def zip_read_pickle_file(zipfile_handler: zipfile.ZipFile,
                         file_name: str) -> object:
    """Read a pickle file from a zip file.

    Args:
        zipfile_handler (zipfile.ZipFile): Zip file object.
        file_name (str): Name of the pickle file.

    Returns:
        object: Pickled data.
    """
    data = zipfile_handler.open(file_name).read()
    data = pickle.loads(data)
    return data


def zip_encode_base64_image(zipfile_handler: zipfile.ZipFile,
                            file_name: str) -> str:
    """Encode a base64 image from a zip file.

    Args:
        zipfile_handler (zipfile.ZipFile): Zip file object.
        file_name (str): Name of the image file.

    Returns:
        str: Base64 encoded image.
    """
    image_content = zipfile_handler.open(file_name).read()
    base64_encoded_image = base64.b64encode(image_content).decode('utf-8')

    return base64_encoded_image


def zip_read_video_and_save(zipfile_handler: zipfile.ZipFile,
                            file_name: str,
                            output_file_name_prefix: str = None) -> str:
    """Read a video file from a zip file and save it to a temporary file.

    Args:
        zipfile_handler (zipfile.ZipFile): Zip file object.
        file_name (str): Name of the video file.
        output_file_name_prefix (str, optional): Prefix of the temporary
            file name. Defaults to None.

    Returns:
        str: Path to the temporary file.
    """
    video_content = zipfile_handler.open(file_name).read()
    file_name_list = file_name.split('/')
    # pre append output_file_name_prefix to the file name list
    if output_file_name_prefix:
        file_name_list = [output_file_name_prefix] + file_name_list
    temp_file_path = '_'.join(file_name_list)
    with open(temp_file_path, 'wb') as f:
        f.write(video_content)
    return temp_file_path
