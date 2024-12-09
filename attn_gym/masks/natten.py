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


def generate_tiled_natten(
    canvas_w: int,
    canvas_h: int,
    kernel_w: int,
    kernel_h: int,
    tile_w: int, 
    tile_h: int,
) -> _mask_mod_signature:
    """Generates a NATTEN attention mask with a given kernel size under static tiled layout. 
    Args:
        canvas_w: The width of the canvas.
        canvas_h: The height of the canvas.
        kernel_w: The width of the kernel.
        kernel_h: The height of the kernel.
        tile_w: The width of the static tile.
        tile_h: The height of the static tile.
    """
    def get_x_y_tiled(idx):
        t_id = idx // (tile_w * tile_h)
        t_x, t_y = t_id // (kernel_w // tile_w), t_id % (kernel_w // tile_w)
        t_offset = idx % (tile_h * tile_w)
        i_x, i_y = t_offset // tile_w, t_offset % tile_w
        return t_x*tile_w + i_x, t_y*tile_h + i_y
    def natten_mask_mod(
        b: IntTensor,
        h: IntTensor,
        q_idx: IntTensor,
        kv_idx: IntTensor,
    ) -> BoolTensor:
        q_x, q_y = get_x_y_tiled(q_idx)
        kv_x, kv_y = get_x_y_tiled(kv_idx)
        # kernel nominally attempts to center itself on the query, but kernel center
        # is clamped to a fixed distance (kernel half-length) from the canvas edge
        kernel_center_x = q_x.clamp(kernel_w // 2, (canvas_w - 1) - kernel_w // 2)
        kernel_center_y = q_y.clamp(kernel_h // 2, (canvas_h - 1) - kernel_h // 2)
        hori_mask = (kernel_center_x - kv_x).abs() <= kernel_w // 2
        vert_mask = (kernel_center_y - kv_y).abs() <= kernel_h // 2
        return hori_mask & vert_mask

    natten_mask_mod.__name__ = f"tiled_natten_c{canvas_w}x{canvas_h}_k{kernel_w}x{kernel_h}_t{tile_w}x{tile_h}"
    return natten_mask_mod


def interleave_bits_32(x):
    """
    Interleave the bits of a 16-bit integer x, producing a 32-bit integer
    where the bits of x are interleaved with zeros.
    """
    x &= 0x0000FFFF  # Ensure x is 16 bits
    x = (x | (x << 8)) & 0x00FF00FF
    x = (x | (x << 4)) & 0x0F0F0F0F
    x = (x | (x << 2)) & 0x33333333
    x = (x | (x << 1)) & 0x55555555
    return x

def morton_encode(x, y):
    """
    Encode 2D coordinates (x, y) into a Morton code (Z-order curve index).

    Parameters:
    x (int): The x-coordinate.
    y (int): The y-coordinate.

    Returns:
    int: The Morton code resulting from interleaving the bits of x and y.
    """
    return (interleave_bits_32(y) << 1) | interleave_bits_32(x)

def deinterleave_bits_32(code):
    """
    Deinterleave bits to retrieve the original 16-bit integer.
    """
    code &= 0x55555555
    code = (code | (code >> 1)) & 0x33333333
    code = (code | (code >> 2)) & 0x0F0F0F0F
    code = (code | (code >> 4)) & 0x00FF00FF
    code = (code | (code >> 8)) & 0x0000FFFF
    return code

def morton_decode(code):
    """
    Decode a Morton code to retrieve the original 2D coordinates (x, y).

    Parameters:
    code (int): The Morton code.

    Returns:
    tuple: A tuple (x, y) representing the original coordinates.
    """
    x = deinterleave_bits_32(code)
    y = deinterleave_bits_32(code >> 1)
    return x, y


def generate_morton_natten(
    canvas_w: int,
    canvas_h: int,
    kernel_w: int,
    kernel_h: int,
) -> _mask_mod_signature:
    """Generates a NATTEN attention mask with a given kernel size under morton curve layout. 
    Args:
        canvas_w: The width of the canvas.
        canvas_h: The height of the canvas.
        kernel_w: The width of the kernel.
        kernel_h: The height of the kernel.
    """
    def natten_mask_mod(
        b: IntTensor,
        h: IntTensor,
        q_idx: IntTensor,
        kv_idx: IntTensor,
    ) -> BoolTensor:
        q_x, q_y = morton_decode(q_idx)
        kv_x, kv_y = morton_decode(kv_idx)
        # kernel nominally attempts to center itself on the query, but kernel center
        # is clamped to a fixed distance (kernel half-length) from the canvas edge
        kernel_center_x = q_x.clamp(kernel_w // 2, (canvas_w - 1) - kernel_w // 2)
        kernel_center_y = q_y.clamp(kernel_h // 2, (canvas_h - 1) - kernel_h // 2)
        hori_mask = (kernel_center_x - kv_x).abs() <= kernel_w // 2
        vert_mask = (kernel_center_y - kv_y).abs() <= kernel_h // 2
        return hori_mask & vert_mask

    natten_mask_mod.__name__ = f"morton_natten_c{canvas_w}x{canvas_h}_k{kernel_w}x{kernel_h}"
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
    
    
    tiled_natten_mask = generate_tiled_natten(
        canvas_w=CANVAS_WIDTH,
        canvas_h=CANVAS_HEIGHT,
        kernel_w=kernel_size,
        kernel_h=kernel_size,
        tile_w=2,
        tile_h=2,
    )
    visualize_attention_scores(
        # TODO: update visualize_attention_scores to support 2D sequences
        query.flatten(start_dim=2, end_dim=3),
        key.flatten(start_dim=2, end_dim=3),
        mask_mod=tiled_natten_mask,
        device=device,
        name=tiled_natten_mask.__name__,
    )
    
    
    morton_natten_mask = generate_morton_natten(
        canvas_w=CANVAS_WIDTH,
        canvas_h=CANVAS_HEIGHT,
        kernel_w=kernel_size,
        kernel_h=kernel_size,
    )
    visualize_attention_scores(
        # TODO: update visualize_attention_scores to support 2D sequences
        query.flatten(start_dim=2, end_dim=3),
        key.flatten(start_dim=2, end_dim=3),
        mask_mod=morton_natten_mask,
        device=device,
        name=morton_natten_mask.__name__,
    )


if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    CLI(main)
