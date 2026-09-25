"""Reproducibility helpers."""

import random

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    """Set random seeds used by this project's local runtime."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
