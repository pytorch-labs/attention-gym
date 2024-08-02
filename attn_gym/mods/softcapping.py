"""Implementation of an tanh softcapping score mod popularized in Gemma2 paper."""

from torch import tanh
from torch.nn.attention.flex_attention import _score_mod_signature


def generate_tanh_softcap(soft_cap: int) -> _score_mod_signature:
    """Returns an tanh bias score_mod given the number of heads H

    Args:
        soft_cap: The soft cap value to use for normalizing logits

    Returns:
        tanh_softcap: score_mod
    """

    def tanh_softcap(score, b, h, q_idx, kv_idx):
        return score * tanh(score / soft_cap)

    return tanh_softcap


def main(device: str = "cpu"):
    """Visualize the attention scores tanh_softcap score mod.

    Args:
        device (str): Device to use for computation. Defaults
    """
    import torch
    from attn_gym import visualize_attention_scores

    B, H, SEQ_LEN, HEAD_DIM = 1, 1, 12, 8

    def make_tensor():
        return torch.rand(B, H, SEQ_LEN, HEAD_DIM, device=device)

    query, key = make_tensor(), make_tensor()

    tanh_softcap_score_mod = generate_tanh_softcap(30)

    visualize_attention_scores(
        query, key, score_mod=tanh_softcap_score_mod, device=device, name="tanh_softcap_score_mod"
    )


if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    CLI(main)
