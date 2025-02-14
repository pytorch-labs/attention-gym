from functools import lru_cache
from typing import Optional, List

import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, DeviceMesh, Partial, Replicate, Shard


from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    create_block_mask,
    flex_attention,
    _mask_mod_signature,
)

from attn_gym.masks.document_mask import length_to_offsets
from attn_gym.masks import (
    causal_mask,
    generate_doc_mask_mod,
)
from attn_gym.load_balance import load_balance_algo


def get_device_type() -> str:
    return "cuda"


@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda"):
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device)
    return block_mask


# TODO: re-write it into a wrapper???
def rewrite_mask_mod_for_cp(
    mask_mod: _mask_mod_signature,
    rank: int,
    block_size: int,
    load_balancer_output: torch.Tensor,
) -> _mask_mod_signature:
    def local_q_idx_to_q_idx(local_q_idx) -> int:
        # calculate local block_idx and block_offset
        local_blk_idx, local_blk_offset = (
            local_q_idx // block_size, local_q_idx % block_size
        )
        current_rank_blk_list = load_balancer_output[rank]
        blk_idx = current_rank_blk_list[local_blk_idx]
        return blk_idx * block_size + local_blk_offset

    return lambda b, h, q_idx, kv_idx: mask_mod(
        b, h, local_q_idx_to_q_idx(q_idx), kv_idx
    )


def run_document_masking(device_mesh, max_seq_len, num_docs):
    # initialize the document lengths
    import random

    random.seed(0)
    torch.cuda.manual_seed(0)

    def generate_random_lengths(total_length, num_documents):
        # Initialize all lengths to 1 to ensure each document has at least one token
        lengths = [1] * num_documents
        remaining_length = total_length - num_documents

        # Randomly distribute the remaining length
        for _ in range(remaining_length):
            index = random.randint(0, num_documents - 1)
            lengths[index] += 1

        return lengths

    lengths = generate_random_lengths(max_seq_len, num_docs)
    offsets = length_to_offsets(lengths, torch.device(f'cuda:{torch.cuda.current_device():d}'))  # TODO: replace with a device mesh call
    document_causal_mask = generate_doc_mask_mod(causal_mask, offsets)
    test_mask_with_load_balance(device_mesh, mask_mod=document_causal_mask, S=max_seq_len)


def test_mask_with_load_balance(
    device_mesh: DeviceMesh,
    mask_mod: Optional[_mask_mod_signature] = None,
    B: int = 16,
    H: int = 16,
    S: int = 8192,
    D: int = 64,
    skip_correctness: bool = False,
    print_mask: bool = True,
    device: str = "cuda",
):
    data_type = torch.float16

    # create block mask
    block_mask = create_block_mask_cached(mask_mod, 1, 1, S, S, device=device)
    block_size = _DEFAULT_SPARSE_BLOCK_SIZE  # TODO: get block size from block mask

    # input initialization
    qkv = [
        torch.rand(
            (B, H, S, D),
            device=device_mesh.device_type,
            dtype=data_type,
            requires_grad=True,
        )
        for _ in range(3)
    ]

    # TODO: input sharding with load-balancing
    # sparsity_info = get_sparsity_info_from_block_mask(block_mask)
    # load_balancer_output = load_balance_algo(sparsity_info)
    cp_mesh_size = device_mesh.size()
    load_balancer_output = load_balance_algo(S, cp_mesh_size, block_size)

    seq_dim = 2
    qkv_dist = [
        distribute_tensor(
            t.detach().clone().requires_grad_(), device_mesh, [
                Shard(seq_dim) if i == 0 else Replicate()
            ]
        )
        for (i, t) in enumerate(qkv)
    ]

    q_local, k_full, v_full = (dt.to_local() for dt in qkv_dist)

    # rewrite `block_mask`
    mask_mod: _mask_mod_signature = block_mask.mask_mod
    cp_rank = device_mesh.get_local_rank()
    cp_mask_mod = rewrite_mask_mod_for_cp(
        mask_mod, cp_rank, block_size, load_balancer_output
    )
    cp_block_mask = create_block_mask_cached(
        cp_mask_mod, B=1, H=1, M=S // cp_mesh_size, N=S, device=device
    )

    # Compile the flex_attention function
    compiled_flex_attention = torch.compile(flex_attention, dynamic=False)

    # TODO: this doesn't address the return_lse=True case
    cp_out = compiled_flex_attention(
        q_local,
        k_full,
        v_full,
        score_mod=None,
        block_mask=cp_block_mask,
    )
    assert isinstance(cp_out, torch.Tensor)

    # unshard
    cp_out_dist = DTensor.from_local(cp_out, device_mesh, [Shard(seq_dim)])
    full_cp_out_dist = cp_out_dist.full_tensor()
    # rearrange
    blk_idx_to_origin = load_balancer_output.view(-1)
    num_chunks = blk_idx_to_origin.numel()
    blk_list_rearranged = [None] * num_chunks
    blk_list = torch.chunk(full_cp_out_dist, num_chunks, dim=seq_dim)
    assert len(blk_list) == num_chunks
    for blk_idx, blk in enumerate(blk_list):
        blk_list_rearranged[blk_idx_to_origin[blk_idx].item()] = blk

    full_cp_out_dist = torch.cat(blk_list_rearranged, dim=seq_dim)

    # local flex attention
    expect_out = flex_attention(*qkv, block_mask=block_mask)
    torch.testing.assert_close(full_cp_out_dist, expect_out, atol=1e-1, rtol=1e-2)


def load_balancing_example(world_size: int, rank: int) -> None:
    device_type = get_device_type()
    device_handle = getattr(torch, device_type, None)
    assert device_handle is not None, f"Unsupported device type: {device_type}"
    num_devices_per_host = device_handle.device_count()
    device_handle.set_device(rank % num_devices_per_host)
    torch._dynamo.config.cache_size_limit = 1000

    # init device mesh
    device_mesh = init_device_mesh(device_type=device_type, mesh_shape=(world_size,))

    run_document_masking(device_mesh, max_seq_len=4096, num_docs=12)
    

if __name__ == "__main__":
    # this script is launched via torchrun which automatically manages ProcessGroup
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # assert world_size == 4  # our example uses 4 worker ranks

    try:
        load_balancing_example(world_size, rank)
    finally:
        dist.barrier()
        dist.destroy_process_group()
