# Attention Gym
Attention Gym is a collection of helpful tools and examples for working with flex-attention

## Overview

This repository aims to provide a playground for experimenting with various attention mechanisms using the FlexAttention API. It includes implementations of different attention variants, performance comparisons, and utility functions to help researchers and developers explore and optimize attention mechanisms in their models.

## Features

- Implementations of various attention mechanisms using FlexAttention
- Utility functions for creating and combining attention masks
- Examples of how to use FlexAttention in real-world scenarios

## Getting Started

### Prerequisites

- PyTorch (version 2.5 or higher)

### Installation

```bash
git clone https://github.com/drisspg/attention-gym.git
cd attention-gym
pip install .
```

## Usage

Here's a quick example of how to use the FlexAttention API with a custom attention mechanism:

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from attn_gym.masks import causal_mask

# Create a causal mask
block_mask: BlockMask = create_block_mask(causal_mask)

# Use FlexAttention with a causal mask modification
output = flex_attention(query, key, value, block_mask=causal_mask)
```

## Examples

Check out the `examples/` directory for more detailed examples of different attention mechanisms and how to implement them using FlexAttention.


## Dev

Install dev requirements
```Bash
pip install -e ".[dev]"
```

Install pre-commit hooks
```Bash
pre-commit install
```
