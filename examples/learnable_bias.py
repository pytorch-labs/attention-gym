import functools
import logging
import torch
import torch.nn.functional as F
import json
import argparse
from torch.nn.attention.flex_attention import flex_attention
from typing import Callable, Dict, List, Tuple, Optional
from enum import Enum, auto
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BiasType(Enum):
    RELATIVE_1D = "relative_1d"
    ABSOLUTE_2D = "absolute_2d"
    HEAD_SPECIFIC = "head_specific"
    BATCH_HEAD = "batch_head"
    MULTIPLICATIVE = "multiplicative"
    LOCAL_WINDOW = "local_window"
    GLOBAL_TOKENS = "global_tokens"
    WEIRD = "weird"
    OFFSET = "offset"


class AttentionTrainer:

    def __init__(
        self,
        batch_size: int = 8,
        num_heads: int = 4,
        seq_length: int = 256,
        head_dim: int = 64,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        window_size: int = 16,
        learning_rate: float = 1e-1,
    ):
        self.B = batch_size
        self.H = num_heads
        self.S = seq_length
        self.D = head_dim
        self.W = window_size
        self.device = device
        self.dtype = dtype
        self.lr = learning_rate
        self.which_bias = torch.tensor(0, device=device)
        self.offset = None

        # Initialize bias generators and functions like in the original
        self.bias_generators = {
            BiasType.RELATIVE_1D: self._generate_relative_1d_bias,
            BiasType.ABSOLUTE_2D: self._generate_absolute_2d_bias,
            BiasType.HEAD_SPECIFIC: self._generate_head_specific_bias,
            BiasType.BATCH_HEAD: self._generate_batch_head_bias,
            BiasType.MULTIPLICATIVE: self._generate_multiplicative_bias,
            BiasType.LOCAL_WINDOW: self._generate_local_window_bias,
            BiasType.GLOBAL_TOKENS: self._generate_global_tokens_bias,
            BiasType.WEIRD: self._generate_weird_bias,
            BiasType.OFFSET: self._generate_offset_bias,
        }

        # Copy the bias application functions from the original
        self.bias_functions = {
            BiasType.RELATIVE_1D: self._apply_relative_1d_bias,
            BiasType.ABSOLUTE_2D: self._apply_absolute_2d_bias,
            BiasType.HEAD_SPECIFIC: self._apply_head_specific_bias,
            BiasType.BATCH_HEAD: self._apply_batch_head_bias,
            BiasType.MULTIPLICATIVE: self._apply_multiplicative_bias,
            BiasType.LOCAL_WINDOW: self._apply_local_window_bias,
            BiasType.GLOBAL_TOKENS: self._apply_global_tokens_bias,
            BiasType.WEIRD: self._apply_weird_bias,
            BiasType.OFFSET: self._apply_offset_bias,
        }

    def _generate_tensor(self, *size):
        return torch.randn(
            *size, device=self.device, dtype=self.dtype, requires_grad=True
        )

        # Bias Generators

    def _generate_relative_1d_bias(self):
        return self._generate_tensor(2 * self.S)

    def _generate_absolute_2d_bias(self):
        return self._generate_tensor(self.S, self.S)

    def _generate_head_specific_bias(self):
        return self._generate_tensor(self.H, self.S, self.S)

    def _generate_batch_head_bias(self):
        return self._generate_tensor(self.B, self.H, self.S, self.S)

    def _generate_multiplicative_bias(self):
        return self._generate_tensor(self.S)

    def _generate_local_window_bias(self):
        return self._generate_tensor(2 * self.W + 1)

    def _generate_learned_pattern_bias(self):
        return self._generate_tensor(self.H, self.D)

    def _generate_global_tokens_bias(self):
        return self._generate_tensor(self.S)

    def _generate_weird_bias(self):
        return self._generate_tensor(self.B, self.H, 4, self.S)

    def _generate_offset_bias(self):
        # Generate both the bias and offset tensors
        bias = self._generate_tensor(self.S)
        self.offset = torch.randint(0, self.S, (self.S,), device=self.device)
        return bias

    # Bias Application Functions
    def _apply_relative_1d_bias(self, score, b, h, q_idx, kv_idx, bias):
        return score + bias[torch.abs(q_idx - kv_idx)]

    def _apply_absolute_2d_bias(self, score, b, h, q_idx, kv_idx, bias):
        return score + bias[q_idx, kv_idx]

    def _apply_head_specific_bias(self, score, b, h, q_idx, kv_idx, bias):
        return score + bias[h, q_idx, kv_idx]

    def _apply_batch_head_bias(self, score, b, h, q_idx, kv_idx, bias):
        return score + bias[b, h, q_idx, kv_idx]

    def _apply_multiplicative_bias(self, score, b, h, q_idx, kv_idx, bias):
        return score * bias[q_idx]

    def _apply_local_window_bias(self, score, b, h, q_idx, kv_idx, bias):
        window_idx = torch.clamp(q_idx - kv_idx + self.W, 0, 2 * self.W)
        return score + bias[window_idx]

    def _apply_global_tokens_bias(self, score, b, h, q_idx, kv_idx, bias):
        return score + bias[kv_idx]

    def _apply_weird_bias(self, score, b, h, q_idx, kv_idx, bias):
        return score + bias[b, h, self.which_bias, q_idx]

    def _apply_offset_bias(self, score, b, h, q_idx, kv_idx, bias):
        return score + bias[self.offset[q_idx]]

    # Copy all the bias generator and application methods from the original class
    # [Previous methods remain the same as in the original code]

    def generate_dummy_data(self, num_samples: int) -> TensorDataset:
        """Generate dummy training data."""
        queries = torch.randn(
            num_samples, self.B, self.H, self.S, self.D, device=self.device
        )
        keys = torch.randn(
            num_samples, self.B, self.H, self.S, self.D, device=self.device
        )
        values = torch.randn(
            num_samples, self.B, self.H, self.S, self.D, device=self.device
        )

        # Generate dummy targets (for this example, we'll try to predict specific patterns)
        targets = torch.randn(
            num_samples, self.B, self.H, self.S, self.D, device=self.device
        )

        return TensorDataset(queries, keys, values, targets)

    def train(
        self,
        bias_type: BiasType = BiasType.RELATIVE_1D,
        num_epochs: int = 10,
        num_samples: int = 2,
        batch_size: int = 4,
    ):
        """Train the attention model with the specified bias type."""
        # Generate bias parameters
        bias = self.bias_generators[bias_type]()
        optimizer = Adam([bias], lr=self.lr)

        # Generate dummy dataset
        dataset = self.generate_dummy_data(num_samples)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Create bias function closure
        def bias_func(score, b, h, q_idx, kv_idx):
            return self.bias_functions[bias_type](score, b, h, q_idx, kv_idx, bias)

        # Compile the attention function
        flex_compiled = torch.compile(
            flex_attention, backend="eager", fullgraph=True, dynamic=False
        )

        # Training loop
        for epoch in range(num_epochs):
            total_loss = 0.0
            with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
                for batch_idx, (q_batch, k_batch, v_batch, targets) in enumerate(pbar):
                    q_batch.requires_grad_()
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = flex_compiled(
                        q_batch[0], k_batch[0], v_batch[0], score_mod=bias_func
                    )

                    # Compute loss (MSE for this example)
                    loss = F.mse_loss(outputs, targets[0])

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    pbar.set_postfix({"loss": f"{loss.item():.6f}"})

            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}")

        return bias, avg_loss


def main(
    bias_type: BiasType = BiasType.RELATIVE_1D,
    num_epochs: int = 10,
    num_samples: int = 2,
    batch_size: int = 4,
):
    trainer = AttentionTrainer()
    trained_bias, final_loss = trainer.train(
        bias_type=bias_type,
        num_epochs=num_epochs,
        num_samples=num_samples,
        batch_size=batch_size,
    )

    logger.info(f"Final loss: {final_loss:.6f}")


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
