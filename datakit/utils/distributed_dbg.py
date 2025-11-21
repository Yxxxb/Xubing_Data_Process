import pdb
import sys
from typing import List

from torch.distributed import barrier, get_rank


class _DistributedPdb(pdb.Pdb):
    """Supports using PDB from inside a multiprocessing child process.

    Usage:
    _DistributedPdb().set_trace()
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def breakpoint(rank: int = 0):
    """Set a breakpoint, but only on a single rank.  All other ranks will wait
    for you to be done with the breakpoint before continuing.

    Args:
        rank (int): Which rank to break on.  Default: ``0``
    """
    if get_rank() == rank:
        pdb = _DistributedPdb()
        pdb.message('\n!!! ATTENTION !!!\n\n'
                    f"Type 'up' to get to the frame \
            that called dist.breakpoint(rank={rank})\n")
        pdb.set_trace()
    barrier()


def count_consecutive_numbers(number_list: List[int],
                              target_number: int) -> List[int]:
    """Count the amount of consecutive numbers in a list.

    Args:
        number_list (List[int]): A list of integers.
        target_number (int): The target number to count.

    Returns:
        List[int]: A list of integers representing the amount of consecutive
            numbers in the list.
    """
    counts = []
    count = 0

    for num in number_list:
        if num == target_number:
            count += 1
        else:
            if count > 0:
                counts.append(count)
                count = 0
    if count > 0:
        counts.append(count)

    return counts
