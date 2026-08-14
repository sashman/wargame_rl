from functools import lru_cache
from typing import TypeAlias

import torch
from loguru import logger

Device: TypeAlias = str | None | torch.device


def _cuda_is_usable() -> bool:
    """Whether this torch build can actually run kernels on the visible GPU.

    `torch.cuda.is_available()` answers "is there a CUDA device", not "can I use
    it": a card whose compute capability predates the build reports available,
    accepts `.to("cuda")`, and then fails on the first forward with `no kernel
    image is available for execution on the device`. Measured here on a
    GTX 1080 Ti (`sm_61`) against a build listing `sm_70`+ — every checkpoint
    path (`simulate`, `just debug`, `measure-checkpoint`) died at the first
    inference rather than at startup, which reads as a bug in whatever was being
    run rather than in the machine.

    Torch warns about the mismatch at import and then proceeds anyway, so the
    warning is not a guard. This is.
    """
    if not torch.cuda.is_available():
        return False
    arches = torch.cuda.get_arch_list()
    if not arches:
        # A build with no arch list is a CPU-only or source build; trust the
        # availability check rather than refusing on missing metadata.
        return True
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}" in arches


@lru_cache(maxsize=1)
def auto_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon (Metal backend)
    if _cuda_is_usable():
        return torch.device("cuda")
    if torch.cuda.is_available():
        logger.warning(
            f"Falling back to CPU: this GPU is compute capability "
            f"sm_{''.join(map(str, torch.cuda.get_device_capability()))}, and "
            f"this torch build only has kernels for "
            f"{', '.join(torch.cuda.get_arch_list())}."
        )
    return torch.device("cpu")  # Fallback


def get_device(device: Device) -> torch.device:
    if device is None:
        return auto_device()
    elif isinstance(device, str):
        return torch.device(device)
    else:
        return device
