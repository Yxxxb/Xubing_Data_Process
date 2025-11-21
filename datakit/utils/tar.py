import base64
import json
import os
import tarfile
from typing import Tuple


def read_all_files_in_tar(tar_file_path: str,
                          extension: str = '.jpg',
                          mode: str = 'r') -> Tuple[tarfile.TarFile, dict]:
    """Read all files in a tar file.

    Used to read webdataset-style tar files.

    Warning: do not forget to close the tarfile object after using it.

    Args:
        tar_file_path (str): The path of the tar file.
        extension (str, optional): The extension of files to return.
            Defaults to '.jpg'.

    Returns:
        A tuple of two values:
        - The tarfile object.
        - A dictionary of all files in the tar file, where the keys are
          the file names and the values are the corresponding TarInfo objects.
    """

    tar = tarfile.open(tar_file_path, mode=mode)
    files_dict = {
        member.name: member
        for member in tar.getmembers()
        if (member.isfile() and member.name.endswith(extension))
        or extension == 'any'
    }
    return tar, files_dict


def tar_encode_base64_image(tar_handle: tarfile.TarFile,
                            member: tarfile.TarInfo) -> str:
    """Encode an image file in a tar file as base64.

    Args:
        tar_handle (tarfile.TarFile): The tar file handle.
        member (tarfile.TarInfo): The TarInfo object of the image file.

    Returns:
        The base64-encoded string of the image file.
    """
    image_content = tar_handle.extractfile(member).read()
    base64_encoded_image = base64.b64encode(image_content).decode('utf-8')
    return base64_encoded_image


def tar_read_video_and_save(tar_handle: tarfile.TarFile,
                            member: tarfile.TarInfo,
                            file_name: str,
                            output_file_name_prefix: str = None) -> str:
    """Read a video file in a tar file and save it to a temporary file.

    Args:
        tar_handle (tarfile.TarFile): The tar file handle.
        member (tarfile.TarInfo): The TarInfo object of the video file.
        file_name (str): The name of the video file.
        output_file_name_prefix (str, optional): The prefix of the
            output file name. Defaults to None.

    Returns:
        The path of the saved video file.
    """
    video_content = tar_handle.extractfile(member).read()
    temp_file_path = output_file_name_prefix + '_' + file_name
    with open(temp_file_path, 'wb') as f:
        f.write(video_content)
    return temp_file_path


def tar_read_json_file(tar_handle: tarfile.TarFile,
                       member: tarfile.TarInfo) -> dict:
    """Read a JSON file in a tar file.

    Args:
        tar_handle (tarfile.TarFile): The tar file handle.
        member (tarfile.TarInfo): The TarInfo object of the JSON file.


    Returns:
        The dictionary of the JSON file.
    """
    json_data = tar_handle.extractfile(member).read()
    json_dict = json.loads(json_data)
    return json_dict


def tar_read_jsonl_file(tar_handle: tarfile.TarFile,
                        member: tarfile.TarInfo) -> dict:
    """Read a JSONL file in a tar file.

    Args:
        tar_handle (tarfile.TarFile): The tar file handle.
        member (tarfile.TarInfo): The TarInfo object of the JSONL file.

    Returns:
        The list of the JSONL file.
    """
    jsonl_file = tar_handle.extractfile(member).readlines()
    jsonl_list = [json.loads(line.strip()) for line in jsonl_file]
    return jsonl_list


def inspect_tar_file(inspect_path: str = '/mnt/cephfs/bensenliu') -> None:
    """Inspect a tar file.

    Args:
        inspect_path (str): The path of saving the inspection result.
            Defaults to '/mnt/cephfs/bensenliu'.
    """

    tar_file = input('Enter the path of the tar file: ')
    tar, files_dict = read_all_files_in_tar(tar_file, extension='any')
    image_path = None
    proceed = input('Do you want to inspect the tar file? (y/n) ')
    for name, member in files_dict.items():
        if not name.endswith('.jpg'):
            continue
        if proceed.lower() != 'y':
            print('Exiting...')
            break
        if image_path is not None:
            os.remove(image_path)
            image_path = None
        image_path = os.path.join(inspect_path, name)
        tar.extract(member, path=inspect_path)
        json_name = name.replace('.jpg', '.json')
        json_data = tar.extractfile(files_dict[json_name]).read()
        json_dict = json.loads(json_data)
        caption = json_dict['caption']
        print(f'Caption: {caption}')
        status = json_dict['status']
        print(f'Status: {status}')
        proceed = input('Do you want to inspect the tar file? (y/n) ')
    tar.close()
