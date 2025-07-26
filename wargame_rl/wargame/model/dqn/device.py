from functools import lru_cache
from typing import TypeAlias
import torch


Device: TypeAlias = str | None | torch.device


@lru_cache(maxsize=1)
def auto_device() -> torch.device:
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon (Metal backend)
    elif torch.cuda.is_available():
        device = torch.device(
            "cuda"
        )  # Not available on Mac, but included for completeness
    else:
        device = torch.device("cpu")  # Fallback
    return device


def get_device(device: Device) -> torch.device:
    if device is None:
        return auto_device()
    elif isinstance(device, str):
        return torch.device(device)
    else:
        return device
