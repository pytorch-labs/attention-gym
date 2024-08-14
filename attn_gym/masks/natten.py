"""Generates a NATTEN mask"""

import torch
from torch import IntTensor, BoolTensor
from torch.nn.attention.flex_attention import _mask_mod_signature
from typing import Tuple


def generate_natten(
    canvas_w: int,
    canvas_h: int,
    kernel_w: int,
    kernel_h: int,
) -> _mask_mod_signature:
    """Generates a NATTEN attention mask with a given kernel size.
    Args:
        canvas_w: The width of the canvas.
        canvas_h: The height of the canvas.
        kernel_w: The width of the kernel.
        kernel_h: The height of the kernel.
    """

    def get_x_y(idx: IntTensor) -> Tuple[IntTensor, IntTensor]:
        return idx // canvas_w, idx % canvas_w

    def natten_mask_mod(
        b: IntTensor,
        h: IntTensor,
        q_idx: IntTensor,
        kv_idx: IntTensor,
    ) -> BoolTensor:
        q_x, q_y = get_x_y(q_idx)
        kv_x, kv_y = get_x_y(kv_idx)
        # kernel nominally attempts to center itself on the query, but kernel center
        # is clamped to a fixed distance (kernel half-length) from the canvas edge
        kernel_center_x = q_x.clamp(kernel_w // 2, (canvas_w - 1) - kernel_w // 2)
        kernel_center_y = q_y.clamp(kernel_h // 2, (canvas_h - 1) - kernel_h // 2)
        hori_mask = (kernel_center_x - kv_x).abs() <= kernel_w // 2
        vert_mask = (kernel_center_y - kv_y).abs() <= kernel_h // 2
        return hori_mask & vert_mask

    natten_mask_mod.__name__ = f"natten_c{canvas_w}x{canvas_h}_k{kernel_w}x{kernel_h}"
    return natten_mask_mod


def main(device: str = "cpu"):
    """Visualize the attention scores of NATTEN mask mod.
    Note: a more complete implementation of NATTEN would include support for kernel dilation.
    The NATTEN unfused kernel also has features like the ability to cross-attend to register tokens.
    This capability is possible to express in Flex Attention but not attempted here.
    See https://github.com/SHI-Labs/NATTEN for more details.

    Args:
        device (str): Device to use for computation. Defaults
    """
    from attn_gym import visualize_attention_scores

    B, H, CANVAS_HEIGHT, CANVAS_WIDTH, HEAD_DIM = 1, 1, 6, 6, 8

    def make_tensor():
        return torch.ones(B, H, CANVAS_HEIGHT, CANVAS_WIDTH, HEAD_DIM, device=device)

    query, key = make_tensor(), make_tensor()

    kernel_size = 3
    natten_mask = generate_natten(
        canvas_w=CANVAS_WIDTH,
        canvas_h=CANVAS_HEIGHT,
        kernel_w=kernel_size,
        kernel_h=kernel_size,
    )
    visualize_attention_scores(
        # TODO: update visualize_attention_scores to support 2D sequences
        query.flatten(start_dim=2, end_dim=3),
        key.flatten(start_dim=2, end_dim=3),
        mask_mod=natten_mask,
        device=device,
        name=natten_mask.__name__,
    )


if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    CLI(main)
