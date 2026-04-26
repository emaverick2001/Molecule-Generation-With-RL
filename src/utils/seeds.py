# Seed utilities for reproducible experiment runs.

# Use this module at the start of every pipeline entrypoint (baseline, rerank,
# reward-filtering, post-training) to reduce randomness across runs.

# Main API:
# - `set_seed(seed: int) -> None`

# What `set_seed` does:
# - Always seeds Python's `random`.
# - If NumPy is installed, seeds NumPy RNG.
# - If PyTorch is installed, seeds CPU and CUDA RNGs.
# - Enables best-effort deterministic CuDNN behavior.

# Notes:
# - Call once right after loading config (before model/data initialization).
# - Determinism can reduce throughput on GPU.
# - If NumPy/PyTorch are not installed, function still works (best-effort mode).

from __future__ import annotations

import random


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducible MVP experiments.

    Always sets:
    - Python random seed

    If installed, also sets:
    - NumPy seed
    - PyTorch CPU/CUDA seeds
    """
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Best-effort deterministic behavior.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    except ImportError:
        pass