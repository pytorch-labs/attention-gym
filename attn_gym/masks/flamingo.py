"""Mask Mod for VisionCrossAttention from the 🦩 Flamingo Paper: https://arxiv.org/pdf/2204.14198"""

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import _mask_mod_signature


def generate_vision_cross_attention_mask_mod(
    intervals: Tensor,
    image_token_length: int,
) -> _mask_mod_signature:
    """
    Generates a mask mod for VisionCrossAttention.

    Args:
        intervals: Tensor of shape (num_images, 2) containing the start and end indices for each image.
        image_token_length: Number of tokens per image.
    """
    num_images = intervals.shape[0]
    image_boundaries = torch.repeat_interleave(
        torch.arange(num_images, device=intervals.device), repeats=image_token_length
    )

    def vision_cross_attention_mask_mod(b, h, q_idx, kv_idx):
        image_idx = image_boundaries[kv_idx]
        interval = intervals[image_idx]
        return (q_idx >= interval[0]) & (q_idx < interval[1])

    return vision_cross_attention_mask_mod


def main(device: str = "cpu"):
    """
    Demonstrate the usage of the VisionCrossAttention mask mod.

    In this case we would generate a mask of
    12 x sum(image_tokens_1 + image_tokens_2 + image_tokens_3)

    assuming image_tokens are size 3

            img1    img2   img3
        1   █ █ █ | ░ ░ ░ | ░ ░ ░
        1   █ █ █ | █ █ █ | ░ ░ ░
     9673   █ █ █ | █ █ █ | ░ ░ ░
      527   █ █ █ | █ █ █ | ░ ░ ░
     1403   █ █ █ | █ █ █ | ░ ░ ░
    12875   █ █ █ | █ █ █ | ░ ░ ░
       13   █ █ █ | █ █ █ | ░ ░ ░
        1   ░ ░ ░ | ░ ░ ░ | █ █ █
     1115   ░ ░ ░ | ░ ░ ░ | █ █ █
      374   ░ ░ ░ | ░ ░ ░ | █ █ █
      264   ░ ░ ░ | ░ ░ ░ | █ █ █
     8415   ░ ░ ░ | ░ ░ ░ | █ █ █

    ```
    """
    from attn_gym import visualize_attention_scores

    num_text_tokens = 12
    num_images = 3
    image_token_length = 3
    intervals = torch.tensor([[0, 7], [1, 7], [7, 12]], dtype=torch.int32, device=device)

    B, H, HEAD_DIM = 1, 1, 8

    def make_tensor(seq_len):
        return torch.ones(B, H, seq_len, HEAD_DIM, device=device)

    query, key = make_tensor(num_text_tokens), make_tensor(num_images * image_token_length)

    vision_cross_attention_mask = generate_vision_cross_attention_mask_mod(
        intervals, image_token_length
    )

    visualize_attention_scores(
        query,
        key,
        mask_mod=vision_cross_attention_mask,
        device=device,
        name="vision_cross_attention_mask",
    )


if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .[viz]")
    CLI(main)
