"""Device selection helpers."""

import torch


def mps_available() -> bool:
    """Return whether PyTorch can use Apple's Metal backend."""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def select_device() -> torch.device:
    """Prefer CUDA, then MPS, and fall back to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_available():
        return torch.device("mps")
    return torch.device("cpu")
