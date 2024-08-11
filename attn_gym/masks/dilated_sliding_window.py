import torch
from torch.nn.attention.flex_attention import _mask_mod_signature


def generate_dilated_sliding_window(window_size: int, dilation: int) -> _mask_mod_signature:
    """Generates a dilated sliding window attention mask.
    Args:
        window_size: The size of the sliding window.
        dilation: The dilation factor for the sliding window.

    Note:
        Query at position i can only attend to keys within a window of size `window_size`
        centered around i, where the keys are at positions j such that:
        * abs(i - j) <= window_size
        * abs(i - j) % dilation == 0
    """

    def dilated_sliding_window(b, h, q_idx, kv_idx):
        diff = torch.abs(q_idx - kv_idx)
        in_window = diff <= window_size
        is_dilated = (diff % dilation) == 0
        return in_window & is_dilated

    dilated_sliding_window.__name__ = f"dilated_sliding_window_{window_size}_dilation_{dilation}"
    return dilated_sliding_window


def main(device: str = "cpu"):
    """Visualize the attention scores of dilated sliding window mask mod.

    Args:
        device (str): Device to use for computation.
    """
    from attn_gym import visualize_attention_scores

    B, H, SEQ_LEN, HEAD_DIM = 1, 1, 24, 8

    def make_tensor():
        return torch.ones(B, H, SEQ_LEN, HEAD_DIM, device=device)

    query, key = make_tensor(), make_tensor()

    dilated_sliding_window_mask = generate_dilated_sliding_window(window_size=4, dilation=2)
    visualize_attention_scores(
        query,
        key,
        mask_mod=dilated_sliding_window_mask,
        device=device,
        name="dilated_sliding_window_mask",
    )


if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    CLI(main)
