# Attention Gym - Claude Context

## Repository Overview

Attention Gym is a collection of helpful tools and examples for working with PyTorch's FlexAttention API. This repository provides implementations of various attention mechanisms, performance benchmarks, and utility functions for researchers and developers.

## Project Structure

- `attn_gym/`: Main package containing attention implementations
  - `masks/`: Examples for creating BlockMasks (causal, sliding window, document masks, etc.)
  - `mods/`: Examples for creating score_mods (alibi, latent attention, softcapping)
  - `paged_attention/`: PagedAttention implementations and benchmarks
- `examples/`: End-to-end implementations and benchmarks
- `test/`: Test files for the project

## Development Commands

### Setup
```bash
# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Testing
```bash
# Run tests
pytest

# Run specific test files
pytest test/test_mods.py
pytest test/test_natten.py
```

### Code Quality
```bash
# Run linting
ruff check

# Run formatting
ruff format

# Run pre-commit on all files
pre-commit run --all-files
```

### Running Examples
```bash
# Most files can be executed directly
python attn_gym/masks/document_mask.py
python examples/benchmark.py
python examples/mla.py
```

## Key Dependencies

- PyTorch (>=2.5 for FlexAttention support)
- Optional: matplotlib, numpy (for visualization examples)
- Dev: pytest, ruff, pre-commit

## Package Information

- Package name: `attn_gym`
- Python requirement: >=3.9
- License: BSD 3-Clause
- Line length: 99 characters (ruff config)