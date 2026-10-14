from __future__ import annotations

import os
import random

import numpy as np

# Canonical seeds used across experiments for seed-stability studies.
DEFAULT_SEEDS: tuple[int, ...] = (7, 13, 23, 42, 2025)


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and (if available) Torch for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:  # torch is an optional dependency
        pass
