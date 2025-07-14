import itertools
from dataclasses import dataclass
from typing import List
import importlib
import sys

import torch
from tabulate import tabulate
from tqdm import tqdm
from attn_gym.masks import (
    causal_mask,
    generate_sliding_window,
    generate_prefix_lm_mask,
    generate_doc_mask_mod,
    generate_dilated_sliding_window,
)
from attn_gym.masks.document_mask import generate_random_lengths, length_to_offsets

from torch.nn.attention.flex_attention import create_block_mask, _mask_mod_signature, noop_mask

has_nuggies = importlib.util.find_spec("transformer_nuggets")
if not has_nuggies:
    print(
        "Need to install transformer_nuggets for this benchmark. "
        "Run `pip install git+https://github.com/drisspg/transformer_nuggets`"
    )
    # Exit if the dependency is missing
    sys.exit(1)

from transformer_nuggets.utils import (  # noqa: E402
    max_memory_usage,
    cuda_memory_usage,
    benchmark_cuda_function_in_microseconds_triton,
)

device = torch.device("cuda")

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


MASK_MOD_MAP = {
    "causal": causal_mask,
    "sliding_window": generate_sliding_window,
    "prefix_lm": generate_prefix_lm_mask,
    "doc_mask_mod": generate_doc_mask_mod,
    "dilated_sliding_window": generate_dilated_sliding_window,
}


@dataclass(frozen=True)
class ExperimentConfig:
    B: int
    H: int
    M: int
    N: int
    mask_mod_name: str


@dataclass(frozen=True)
class ExperimentResult:
    creation_time_ms: float
    memory_bytes: int
    max_memory_usage: int


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_mask_mod(c: ExperimentConfig) -> _mask_mod_signature:
    match c.mask_mod_name:
        case "sliding_window":
            return generate_sliding_window(window_size=128)
        case "prefix_lm":
            return generate_prefix_lm_mask(prefix_length=128)
        case "doc_mask_mod":
            lengths = generate_random_lengths(total_length=c.M, num_documents=4)
            offsets = length_to_offsets(lengths, device)
            return generate_doc_mask_mod(mask_mod=noop_mask, offsets=offsets)
        case "dilated_sliding_window":
            return generate_dilated_sliding_window(window_size=128, dilation=2)
        case _:
            return MASK_MOD_MAP[c.mask_mod_name]


def get_configs() -> List[ExperimentConfig]:
    # Define ranges for benchmark parameters
    Bs = [1, 4, 8]
    Hs = [8, 16]
    # Sequence lengths - adjust as needed
    # Using powers of 2 up to a reasonable limit for mask creation
    SeqLens = [1024, 2048, 4096, 8192]
    # Map string names to mask functions
    mask_mods_to_run = list(MASK_MOD_MAP.keys())

    configs = []
    for B, H, S, mask_mod in itertools.product(Bs, Hs, SeqLens, mask_mods_to_run):
        configs.append(
            ExperimentConfig(
                B=B,
                H=H,
                M=S,  # Assuming M=N for simplicity
                N=S,
                mask_mod_name=mask_mod,
            )
        )
    return configs


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    # Find the mask_mod function by name
    assert config.mask_mod_name in MASK_MOD_MAP, f"Mask mod '{config.mask_mod_name}' not found."
    mask_mod_fn = get_mask_mod(config)

    # --- Time Benchmarking ---
    cbm = torch.compile(create_block_mask)
    # Warmup
    for _ in range(10):
        cbm(mask_mod_fn, config.B, config.H, config.M, config.N, device=device)
    torch.cuda.synchronize(device)

    creation_time_us = benchmark_cuda_function_in_microseconds_triton(
        lambda: cbm(mask_mod_fn, config.B, config.H, config.M, config.N, device=device),
    )

    torch.cuda.synchronize(device)

    with cuda_memory_usage() as mem:
        bm = cbm(mask_mod_fn, config.B, config.H, config.M, config.N, device=device)
    del bm

    with max_memory_usage() as max_mem:
        bm = cbm(mask_mod_fn, config.B, config.H, config.M, config.N, device=device)
    del bm

    return ExperimentResult(
        creation_time_ms=creation_time_us / 1000,
        memory_bytes=mem.memory_usage,
        max_memory_usage=max_mem.max_memory,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "B",
        "H",
        "M",
        "N",
        "Mask Mod",
        "Creation Time (ms)",
        "Mask Size Memory (GiB)",
        "Max Construction Memory (GiB)",
    ]
    rows = []
    for experiment in experiments:
        rows.append(
            [
                experiment.config.B,
                experiment.config.H,
                experiment.config.M,
                experiment.config.N,
                experiment.config.mask_mod_name,
                f"{experiment.result.creation_time_ms:.4f}",
                f"{experiment.result.memory_bytes/(1024**3):.4f}",
                f"{experiment.result.max_memory_usage/(1024**3):.4f}",
            ]
        )
    # Sort rows for better readability (e.g., by B, H, M, N)
    rows.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    print(tabulate(rows, headers=headers, tablefmt="grid"))


def main():
    torch.random.manual_seed(123)
    configs = get_configs()
    results = []
    print(f"Running {len(configs)} benchmark configurations...")
    for config in tqdm(configs):
        try:
            result = run_experiment(config)
            results.append(Experiment(config=config, result=result))
        except Exception as e:
            print(f"Failed to run config {config}: {e}")
            # Optionally skip failed configs or handle differently

    # Use Tabulate to print results
    print_results(results)


if __name__ == "__main__":
    main()
