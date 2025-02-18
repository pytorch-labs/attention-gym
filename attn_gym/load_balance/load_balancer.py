from abc import ABC, abstractmethod

import torch


__all__ = ["load_balance_algo"]


def load_balance_algo(S: int, size: int, block_size: int) -> torch.Tensor:
    return HeadTail.gen_load_balance_plan(S, size, block_size)


class LoadAlgorithm(ABC):
    @classmethod
    @abstractmethod
    def gen_load_balance_plan(cls, S: int, size: int, block_size: int) -> torch.Tensor:
        pass


class Noop(LoadAlgorithm):
    @classmethod
    def gen_load_balance_plan(cls, S: int, size: int, block_size: int) -> torch.Tensor:
        total_num_blk = S // block_size
        assert S % (size * block_size) == 0
        local_num_blk = total_num_blk // size
        return torch.arange(total_num_blk, device="cuda").view(size, local_num_blk)


class HeadTail(LoadAlgorithm):
    @classmethod
    def gen_load_balance_plan(cls, S: int, size: int, block_size: int) -> torch.Tensor:
        total_num_blk = S // block_size
        assert S % (size * 2 * block_size) == 0
        local_num_blk_pair = total_num_blk // (size * 2)
        plan_tensor = torch.arange(total_num_blk, device="cuda").view(
            -1, local_num_blk_pair
        )
        return torch.stack(
            (
                plan_tensor[:size],
                plan_tensor[size:].flip(dims=(0,)),
            ),
            dim=1,
        ).view(size, -1)


if __name__ == "__main__":
    print(HeadTail.gen_load_balance_plan(32, 4, 1))
