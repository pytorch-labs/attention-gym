from typing import List

import torch


__all__ = ["load_balance_algo"]


def load_balance_algo(S: int, size: int, block_size: int) -> torch.Tensor:
    total_num_blk = S // block_size
    assert S % (size * total_num_blk) == 0
    local_num_blk = total_num_blk // size
    return torch.arange(total_num_blk, device="cuda").view(size, local_num_blk)
