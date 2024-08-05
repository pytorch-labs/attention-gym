# Attention Gym 💪
Attention Gym is a collection of helpful tools and examples for working with flex-attention

![favorite](https://github.com/user-attachments/assets/3747fd24-1282-4d65-9072-882e55dad0ad)

## Overview

This repository aims to provide a playground for experimenting with various attention mechanisms using the FlexAttention API. It includes implementations of different attention variants, performance comparisons, and utility functions to help researchers and developers explore and optimize attention mechanisms in their models.

## Features

- Implementations of various attention mechanisms using FlexAttention
- Utility functions for creating and combining attention masks
- Examples of how to use FlexAttention in real-world scenarios

## 🚀 Getting Started

### Prerequisites

- PyTorch (version 2.5 or higher)

### Installation

```bash
git clone https://github.com/drisspg/attention-gym.git
cd attention-gym
pip install .
```

## Usage

Here's a quick example of how to use the FlexAttention API with a causal_mask:

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from attn_gym.masks import causal_mask

# Create a causal mask
Q_LEN, KV_LEN = query.size(-2), key.size(-2)
block_mask: BlockMask = create_block_mask(causal_mask, 1, 1, Q_LEN, KV_LEN)

# Use FlexAttention with a causal mask modification
output = flex_attention(query, key, value, block_mask=causal_mask)
```
## 📁 Structure

Attention Gym is organized for easy exploration of attention mechanisms:

### 🔍 Key Locations
- `attn_gym.masks`: Examples creating `BlockMasks`
- `attn_gym.mods`: Examples creating `score_mods`
- `examples/`: Detailed implementations using FlexAttention

### 🏃‍♂️ Running Examples
Files are both importable and runnable. To explore:

1. Run files directly:
   ```Shell
   python attn_gym/masks/document_mask.py
   ```
2. Most files generate visualizations when run.

Check out the `examples` directory for end-to-end examples of using FlexAttention in real-world scenarios.

## Note
Attention Gym is under active development, and we do not currently offer any backward compatibility guarantees. APIs and functionalities may change between versions. We recommend pinning to a specific version in your projects and carefully reviewing changes when upgrading.

## 🛠️ Dev

Install dev requirements
```Bash
pip install -e ".[dev]"
```

Install pre-commit hooks
```Bash
pre-commit install
```


## License
attention-gym is released under the BSD 3-Clause License.
