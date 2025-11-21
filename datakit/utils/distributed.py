import json
import os
import subprocess
import time
from pprint import pprint
from typing import List, Tuple

from datakit.utils.files import dump_dict_to_json_file, get_jsonl_size
from datakit.utils.mp import multi_process_with_append


# def get_distributed_env() -> Tuple[int, int, int]:
#     """Get the distributed environment variables.

#     Returns:
#         Tuple[int, int, int]: world_size, rank, local_rank
#     """
#     world_size = int(os.environ.get('WORLD_SIZE', 1))
#     rank = int(os.environ.get('RANK', 0))
#     local_rank = int(os.environ.get('LOCAL_RANK', 0))

#     return world_size, rank, local_rank

def get_distributed_env(is_cpu=False) -> Tuple[int, int, int]:
    """Get the distributed environment variables.

    Returns:
        Tuple[int, int, int]: world_size, rank, local_rank
    """
    if is_cpu or (os.environ.get('WORLD_SIZE') is None and os.environ.get('RANK') is None):
        world_size = int(os.environ.get("TASK_CNT", 1))
        rank = int(os.environ.get("TASK_INDEX", 0))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        return world_size, rank, local_rank
    
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    return world_size, rank, local_rank
    

def dist_split_files(files: List[str]) -> List[str]:
    """Split files for distributed processing.

    Args:
        files (List[str]): List of files to be split.

    Returns:
        List[str]: List of files for the current rank.
    """
    world_size, rank, _ = get_distributed_env()
    num_files = len(files)
    base_num_files_per_rank = num_files // world_size
    extra_files = num_files % world_size
    if rank < extra_files:
        start_idx = (base_num_files_per_rank + 1) * rank
        end_idx = start_idx + base_num_files_per_rank + 1
    else:
        start_idx = (base_num_files_per_rank +
                     1) * extra_files + base_num_files_per_rank * (rank -
                                                                   extra_files)
        end_idx = start_idx + base_num_files_per_rank

    print(f'Rank {rank} processing files {start_idx} to {end_idx}')
    return files[start_idx:end_idx]


def obtain_files_cur_rank(files: List[str], world_size: int,
                          rank: int) -> List[str]:
    """Obatin files for the current rank.

    Args:
        files (List[str]): List of files to be split.
        world_size (int): Number of processes.
        rank (int): Rank of the current process.

    Returns:
        List[str]: List of files for the current rank.
    """
    num_files = len(files)
    base_num_files_per_rank = num_files // world_size
    extra_files = num_files % world_size
    if rank < extra_files:
        start_idx = (base_num_files_per_rank + 1) * rank
        end_idx = start_idx + base_num_files_per_rank + 1
    else:
        start_idx = (base_num_files_per_rank +
                     1) * extra_files + base_num_files_per_rank * (rank -
                                                                   extra_files)
        end_idx = start_idx + base_num_files_per_rank

    print(f'Rank {rank} processing files {start_idx} to {end_idx}')
    return files[start_idx:end_idx]


def obtain_items_cur_rank(items: List[str], world_size: int,
                          rank: int) -> List[str]:
    """Obatin items for the current rank in a file/list.

    Args:
        items (List[str]): List of items to be split.
        world_size (int): Number of processes.
        rank (int): Rank of the current process.

    Returns:
        List[str]: List of items for the current rank.
    """
    num_items = len(items)
    base_num_items_per_rank = num_items // world_size
    extra_items = num_items % world_size
    if rank < extra_items:
        start_idx = (base_num_items_per_rank + 1) * rank
        end_idx = start_idx + base_num_items_per_rank + 1
    else:
        start_idx = (base_num_items_per_rank +
                     1) * extra_items + base_num_items_per_rank * (rank -
                                                                   extra_items)
        end_idx = start_idx + base_num_items_per_rank

    print(f'Rank {rank} processing items {start_idx} to {end_idx}, sum items num: {num_items}')
    return items[start_idx:end_idx]


def workload_balance_dist_split_jsonls(jsonls: List[str],
                                       num_workers: int = 256
                                       ) -> List[str]:  # noqa
    """Split jsonls for distributed processing considering
        the workload balance.

    Args:
        jsonls (List[str]): List of jsonls to be split.
        num_workers (int, optional): Number of workers.

    Returns:
        List[str]: List of jsonls for the current rank.
    """

    world_size, rank, _ = get_distributed_env()
    jsonls_with_size = multi_process_with_append(get_jsonl_size, jsonls,
                                                 num_workers)
    total_size = sum([size for _, size in jsonls_with_size])
    size_per_rank = total_size // world_size
    jsonls_for_world_size = []
    sizes_for_world_size = []
    jsonls_for_rank = []
    size_for_rank = 0
    for jsonl_with_size in jsonls_with_size:
        jsonl_path, size = jsonl_with_size
        jsonls_for_rank.append(jsonl_path)
        size_for_rank += size
        if size_for_rank >= size_per_rank:
            jsonls_for_world_size.append(jsonls_for_rank)
            sizes_for_world_size.append(size_for_rank)
            jsonls_for_rank = []
            size_for_rank = 0
    if len(jsonls_for_world_size) >= world_size:
        jsonls_for_world_size[-1].extend(jsonls_for_rank)
        sizes_for_world_size[-1] += size_for_rank
    else:
        jsonls_for_world_size.append(jsonls_for_rank)
        sizes_for_world_size.append(size_for_rank)
    jsonls_for_world_size.extend([[]] *
                                 (world_size - len(jsonls_for_world_size)))
    sizes_for_world_size.extend([0] * (world_size - len(sizes_for_world_size)))
    print(
        f'Total size: {total_size}, size cur rank: {sizes_for_world_size[rank]}'  # noqa
    )
    return jsonls_for_world_size[rank]


def is_distributed_tasks_finished(meta_files: List[str]) -> bool:
    """Check if all distributed tasks are finished.

    Args:
        meta_files (List[str]): List of meta files to check.

    Returns:
        bool: True if all distributed tasks are finished,
        False otherwise.
    """
    finished = []
    for meta_file in meta_files:
        if os.path.exists(meta_file):
            finished.append(True)
        else:
            finished.append(False)
    return all(finished)


def barrier_all_processes(task_name: str,
                          data: dict = {},
                          root: str = '/mnt/cephfs/bensenliu/wfs/data_log',
                          dump_result: bool = True,
                          read_data: bool = False) -> None:  # noqa
    """Wait for all processes to reach this point.

    Args:
        task_name (str): Name of the task.
        data (dict, optional): Data to be dumped.
            Defaults to {}.
        root (str, optional): Root directory to store meta files.
            Defaults to '/mnt/cephfs/bensenliu/wfs/data_log'.
        dump_result (bool, optional): Whether to dump the result.
            Defaults to True.
        read_data (bool, optional): Whether to read the data.
            Defaults to False.
    """
    world_size, rank, _ = get_distributed_env()
    output_root = os.path.join(root, task_name)
    os.makedirs(output_root, exist_ok=True)
    meta_file = os.path.join(output_root, f'meta_{rank}.json')
    if dump_result:
        dump_dict_to_json_file(meta_file, data)
    meta_files = [
        os.path.join(output_root, f'meta_{i}.json') for i in range(world_size)
    ]
    while True:
        if is_distributed_tasks_finished(meta_files):
            if read_data:
                status_dict = {}
                for meta_file in meta_files:
                    with open(meta_file, 'r') as f:
                        cur_status_dict = json.load(f)
                    for key, value in cur_status_dict.items():
                        if key not in status_dict:
                            status_dict[key] = [value]
                        else:
                            status_dict[key].append(value)
                pprint(status_dict)
            break
        time.sleep(2)
    print('All distributed tasks are finished.')


def gpu_utilization(num_gpus: int = 8) -> int:
    """To maximize GPU utilization

    Args:
        num_gpus (int, optional): Number of GPUs.
            Defaults to 8.

    Returns:
        int: Process ID of the torchrun command.
    """
    command = [
        'torchrun', '--nproc_per_node', f'{num_gpus}',
        '/mnt/cephfs/bensenliu/exp_runs/loop.py'
    ]
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    pid = process.pid
    return pid


def kill_process(process_pattern: str = 'loop.py') -> None:
    """Kill a process.

    Args:
        pid (int): Process ID to kill.
    """
    os.system(f'pkill -f {process_pattern}')


def delete_folder(
        task_name: str,
        root: str = '/mnt/cephfs/bensenliu/wfs/data_log') -> None:  # noqa
    """Delete a folder.

    Args:
        folder_path (str): Path to the folder to be deleted.
    """
    _, rank, _ = get_distributed_env()
    if rank == 0:
        folder_path = os.path.join(root, task_name)
        if os.path.exists(folder_path):
            os.system(f'rm -rf {folder_path}')
