from typing import List


__all__ = ["load_balance_algo"]


def load_balance_algo(S: int, size: int, block_size: int) -> List[List[int]]:
    assert S % (size * block_size) == 0
    num_local_blk = S // (size * block_size)
    return [
        [
            local_blk_idx + rank * num_local_blk
            for local_blk_idx in range(num_local_blk)
        ]
        for rank in range(size)
    ]
