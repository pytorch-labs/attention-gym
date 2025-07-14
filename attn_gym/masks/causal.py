"""Standard Causal Attention Masking."""

import torch
from torch.nn.attention.flex_attention import BlockMask
from attn_gym.utils import cdiv


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def create_causal_block_mask_fast(
    batch_size: int | None,
    num_heads: int | None,
    q_seq_len: int,
    kv_seq_len: int,
    device: torch.device,
    block_size: int = 128,
    separate_full_blocks: bool = True,
) -> BlockMask:
    """Create a causal block mask efficiently without materializing the full mask.

    This function generates the block mask data structure directly for causal attention,
    avoiding the need to create and process a full dense mask. This is much more efficient
    for long sequences.

    Args:
        q_seq_len: Query sequence length
        kv_seq_len: Key/value sequence length
        device: Device to create tensors on
        batch_size: Batch size (defaults to 1 if None)
        num_heads: Number of attention heads (defaults to 1 if None)
        block_size: Block size for the block mask (both Q and KV use same size)
        separate_full_blocks: Whether to separate full blocks from partial blocks

    Returns:
        BlockMask: Block mask object for causal attention
    """
    if batch_size is None:
        batch_size = 1
    if num_heads is None:
        num_heads = 1
    if isinstance(block_size, tuple):
        q_block_size, kv_block_size = block_size
    else:
        q_block_size = kv_block_size = block_size

    num_q_blocks = cdiv(q_seq_len, q_block_size)
    num_kv_blocks = cdiv(kv_seq_len, kv_block_size)

    # For causal mask, each query block can attend to all KV blocks up to and including
    # the diagonal block
    kv_num_blocks = torch.zeros(
        (batch_size, num_heads, num_q_blocks), dtype=torch.int32, device=device
    )
    kv_indices = torch.zeros(
        (batch_size, num_heads, num_q_blocks, num_kv_blocks), dtype=torch.int32, device=device
    )

    if separate_full_blocks:
        full_kv_num_blocks = torch.zeros(
            (batch_size, num_heads, num_q_blocks), dtype=torch.int32, device=device
        )
        full_kv_indices = torch.zeros(
            (batch_size, num_heads, num_q_blocks, num_kv_blocks), dtype=torch.int32, device=device
        )
    else:
        full_kv_num_blocks = None
        full_kv_indices = None

    for q_block_idx in range(num_q_blocks):
        # For causal attention, query block i can attend to KV blocks [0, i]
        # The last block in the diagonal may be partial
        num_full_blocks = min(q_block_idx, num_kv_blocks)
        num_partial_blocks = 1 if q_block_idx < num_kv_blocks else 0

        if separate_full_blocks:
            assert full_kv_num_blocks is not None
            assert full_kv_indices is not None

            if num_partial_blocks > 0:
                min_q_index = q_block_idx * q_block_size
                max_kv_index = min((q_block_idx + 1) * kv_block_size - 1, kv_seq_len - 1)
                is_diagonal_full = min_q_index >= max_kv_index
            else:
                is_diagonal_full = False

            if is_diagonal_full:
                # Diagonal block is full - move it from partial to full
                full_kv_num_blocks[:, :, q_block_idx] = num_full_blocks + 1
                kv_num_blocks[:, :, q_block_idx] = 0  # No partial blocks

                # Set indices for all full blocks (including diagonal)
                indices = torch.arange(num_full_blocks + 1, device=device)
                full_kv_indices[:, :, q_block_idx, : num_full_blocks + 1] = indices
            else:
                # Diagonal block is partial (or doesn't exist)
                full_kv_num_blocks[:, :, q_block_idx] = num_full_blocks
                kv_num_blocks[:, :, q_block_idx] = num_partial_blocks

                # Set indices for full blocks
                if num_full_blocks > 0:
                    indices = torch.arange(num_full_blocks, device=device)
                    full_kv_indices[:, :, q_block_idx, :num_full_blocks] = indices

                # Set index for partial block
                if num_partial_blocks > 0:
                    kv_indices[:, :, q_block_idx, 0] = q_block_idx
        else:
            # All blocks go into partial
            total_blocks = num_full_blocks + num_partial_blocks
            kv_num_blocks[:, :, q_block_idx] = total_blocks
            if total_blocks > 0:
                indices = torch.arange(total_blocks, device=device)
                kv_indices[:, :, q_block_idx, :total_blocks] = indices

    return BlockMask.from_kv_blocks(
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        full_kv_num_blocks=full_kv_num_blocks,
        full_kv_indices=full_kv_indices,
        BLOCK_SIZE=(q_block_size, kv_block_size),
        mask_mod=causal_mask,
        seq_lengths=(q_seq_len, kv_seq_len),
    )


def main(device: str = "cpu"):
    """Visualize the attention scores of causal masking.

    Args:
        device (str): Device to use for computation. Defaults
    """
    import torch
    from attn_gym import visualize_attention_scores

    B, H, SEQ_LEN, HEAD_DIM = 1, 1, 12, 8

    def make_tensor():
        return torch.ones(B, H, SEQ_LEN, HEAD_DIM, device=device)

    query, key = make_tensor(), make_tensor()

    visualize_attention_scores(query, key, mask_mod=causal_mask, device=device, name="causal_mask")


if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    CLI(main)
