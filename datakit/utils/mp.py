import multiprocessing as mp
from typing import List

from tqdm import tqdm


def multi_process_with_extend(func: any,
                              inputs: List[any],
                              num_workers: int = 1) -> List[any]:
    """Multi-process with progress bar and extend the results.

    Args:
        func (any): The function to be executed in parallel.
        inputs (List[any]): The inputs for the function.
        num_workers (int, optional): The number of workers to use.
            Defaults to 1.

    Returns:
        List[any]: The results of the function.
    """
    pool = mp.Pool(num_workers)
    results = []
    with tqdm(total=len(inputs)) as pbar:
        for result in pool.imap_unordered(func, inputs):
            if result is not None:
                results.extend(result)
            pbar.update(1)
    pool.close()
    pool.join()
    return results


def multi_process_with_append(func: any,
                              inputs: List[any],
                              num_workers: int = 1) -> List[any]:
    """Multi-process with progress bar and append the results.

    Args:
        func (any): The function to be executed in parallel.
        inputs (List[any]): The inputs for the function.
        num_workers (int, optional): The number of workers to use.
            Defaults to 1.

    Returns:
        List[any]: The results of the function.
    """
    pool = mp.Pool(num_workers)
    results = []
    with tqdm(total=len(inputs)) as pbar:
        for result in pool.imap_unordered(func, inputs):
            if result is not None:
                results.append(result)
            pbar.update(1)
    pool.close()
    pool.join()
    return results
