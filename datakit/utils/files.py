import json
import os
from typing import List, Optional, Tuple

import pandas as pd

from .mp import multi_process_with_append


def find_all_files(
        root_dir: str,
        extension: Optional[str | Tuple[str] | List[str]] = 'tar') -> list:
    """Find all files with the given extension in the given directory and its
    subdirectories.

    Args:
        root_dir (str): The root directory to search in.
        extension Optional(str|Tuple[str]):
            The extension of the files to search for. Defaults to 'tar'.

    Returns:
        list: A list of all files with the given extension in the
            given directory. The full path will be included.
    """
    files = []
    if isinstance(extension, str):
        extension = (extension, )
    else:
        extension = tuple(extension)
    for root, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(extension):
                files.append(os.path.join(root, filename))
    return files


def find_all_files_multi_folders(
        root_dir: List[str],
        extension: Optional[str | Tuple[str] | List[str]] = 'tar') -> list:
    all_files = []
    for root in root_dir:
        files = find_all_files(root, extension)
        all_files.extend(files)
    return all_files


def remove_single_file(file_path: str) -> None:
    """Remove a single file.

    Args:
        file_path (str): The path of the file to be removed.

    Returns:
        None: None.
    """
    os.remove(file_path)
    return None


def remove_all_files(folder_path: str,
                     num_workers: int = 1,
                     extension: str = 'tar') -> None:
    """Remove all files in a folder.

    Args:
        folder_path (str): The path of the folder to be removed.
        num_workers (int, optional): The number of workers to use for parallel
            processing. Defaults to 1.
        extension (str, optional): The extension of the files to remove.
            Defaults to 'tar'.


    Returns:
        None: None.
    """
    # find all files in the folder recursively
    all_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if extension in file or extension == 'all':
                all_files.append(os.path.join(root, file))
    # remove all files in parallel
    multi_process_with_append(remove_single_file, all_files, num_workers)

    return None


def read_jsonl_file(file_path: str) -> List[any]:
    """Read a JSONL file and return a list of objects.

    Args:
        file_path (str): The path of the JSONL file to be read.

    Returns:
        List[any]: A list of objects read from the JSONL file.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        objects = [json.loads(line.strip()) for line in lines]
    return objects


def mem_efficient_read_jsonl_file(jsonl_file: str):
    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_parquet_file(file_path: str) -> List[any]:
    """Read a parquet file and return a list of objects.

    Args:
        file_path (str): The path of the parquet file to be read.

    Returns:
        List[any]: A list of objects read from the parquet file.
    """
    df = pd.read_parquet(file_path)
    d_list = df.to_dict('records')
    return d_list


def remove_existing_files(input_files: List[str],
                          existing_files: List[str],
                          only_file_name: bool = False) -> List[str]:
    """Remove existing files from the input list.

    Args:
        input_files (List[str]): A list of input files.
        existing_files (List[str]): A list of existing files.

    Returns:
        List[str]: A list of input files without the existing files.
    """
    print(f'=> Number of input files: {len(input_files)}')
    existing_files_dict = dict()
    for existing_file in existing_files:
        existing_folder_name = existing_file.split('/')[-2]
        existing_file_name = existing_file.split('/')[-1]
        if only_file_name:
            key = existing_file_name
        else:
            key = f'{existing_folder_name}/{existing_file_name}'
        existing_files_dict[key] = existing_file

    remaining_files = []
    for input_file in input_files:
        input_folder_name = input_file.split('/')[-2]
        input_file_name = input_file.split('/')[-1]
        if only_file_name:
            key = input_file_name
        else:
            key = f'{input_folder_name}/{input_file_name}'
        if key not in existing_files_dict:
            remaining_files.append(input_file)
    print(f'=> Number of remaining files: {len(remaining_files)}')

    return remaining_files


def dump_list_to_jsonl_file(file_path: str, data: List[any]) -> None:
    """Dump a list of objects to a JSONL file.

    Args:
        file_path (str): The path of the JSONL file to be written.
        data (List[any]): A list of objects to be written to the JSONL file.

    Returns:
        None: None.
    """

    with open(file_path, 'w') as f:
        for item in data:
            json_str = json.dumps(item)
            f.write(json_str + '\n')

    return None


def filterout_repeat_images_for_mmq(results: List[dict]) -> List[dict]:
    """Filter out repeat images for MMQ.

    Args:
        results (List[dict]): A list of image and conversation results.

    Returns:
        List[dict]: A list of image and conversation results
            without repeat images.
    """
    filtered_results = []
    image_ids = dict()
    for result in results:
        if result['type'] == 'conversation':
            filtered_results.append(result)
        else:
            if result['id'] not in image_ids:
                image_ids[result['id']] = True
                filtered_results.append(result)
    return filtered_results


def remove_path_prefix(path: str, prefix: str) -> str:
    """Remove path prefix
    Args:
        path (str): source path
        prefix (str): the prefix path you want to remove

    Returns:
        str: A str of path without prefix
    """
    prefix = os.path.normpath(prefix) + os.path.sep
    if path.startswith(prefix):
        path = path[len(prefix):]
    else:
        print(f'Path {path} not starts with {prefix}')
    return path


def read_mmq_recordio(path: str):
    """Read mimikyu recordio files."""
    from mmq_io.reader import Reader

    assert path.endswith('recordio')
    reader = Reader(path)
    return reader


def read_mmq_index(path: str):
    """Read mimikyu index files."""
    from mmq.data.batch_file_reader import BatchIndexFileReader

    assert (
        path.endswith('recordio') and 'index' in path
    ), "You should only input index files with 'index' and endswith 'recordio'"
    reader = BatchIndexFileReader(path)
    return reader


def read_mmq_index_datasets(path: str):
    """Read mimikyu index files, extract dataset names only."""
    index_reader = read_mmq_index(path)
    filenames = index_reader.header.filenames
    candidate_dataset_names = ('ai2d', 'docvqa', 'gpt4v')

    def match_dataset_prefix_suffix():
        prefix = suffix = None
        for fname in filenames:
            for cand_dataset in candidate_dataset_names:
                if cand_dataset in fname:
                    prefix, suffix = fname.split(cand_dataset)
                    return prefix, suffix
        return prefix, suffix

    prefix, suffix = match_dataset_prefix_suffix()
    if prefix is None or suffix is None:
        print(
            f'Index does not contain any dataset in {candidate_dataset_names}')
        return None

    striped_datasets = [
        fname[len(prefix):][:-len(suffix)] for fname in filenames
    ]
    return striped_datasets


def dump_results_to_txt(results: str, output_file: str) -> None:
    """Dump results to txt file.

    Args:
        results (str): A string of results.
        output_file (str): The path of the output file.

    Returns:
        None: None.
    """
    with open(output_file, 'w') as f:
        f.write(results)


def convert_beautiful_html_to_plain_text(html: str) -> str:
    """Convert beautiful html to plain text.

    Args:
        html (str): A string of html.

    Returns:
        str: A string of plain text.
    """
    lines = html.split('\n')
    new_lines = []
    for line in lines:
        line = line.strip()
        new_lines.append(line)
    return ''.join(new_lines)


def dump_dict_to_json_file(output_file: str, data: dict = {}) -> None:
    """Dump a dictionary to a JSON file.

    Args:
        data (dict, optional): A dictionary to be written to the JSON file.
            Defaults to {}.
        output_file (str): The path of the output file.

    Returns:
        None: None.
    """
    with open(output_file, 'w') as f:
        json.dump(data, f)


def get_jsonl_size(jsonl_path: str) -> Tuple[str, int]:
    """Get the size of a JSONL file.

    Args:
        jsonl_path (str): The path of the JSONL file.

    Returns:
        int: The size of the JSONL file.
    """
    try:
        data = read_jsonl_file(jsonl_path)
        return (jsonl_path, len(data))
    except Exception as e:
        print(f'Error reading {jsonl_path}: {e}')
        return None
