"""Device detection and dtype selection."""

import torch


def get_device() -> str:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype(device: str) -> torch.dtype:
    """Get appropriate dtype for device.

    MPS -> float16, CUDA -> bfloat16, CPU -> float32.
    """
    if device == "mps":
        return torch.float16
    elif device == "cuda":
        return torch.bfloat16
    return torch.float32
