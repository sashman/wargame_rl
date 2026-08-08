"""Device-level performance settings, applied once at startup.

Kept out of the Lightning modules deliberately: these are process-wide torch
settings, not model configuration, and applying them from a module constructor
would make them depend on how many models a process happens to build.
"""

from __future__ import annotations

import torch
from loguru import logger

# TF32 tensor cores arrived with Ampere. Below sm_80 the setting is accepted and
# silently does nothing, so the capability is checked rather than the flag.
_TF32_MIN_CAPABILITY = (8, 0)


def tf32_is_available() -> bool:
    """Whether the current CUDA device has TF32 tensor cores."""
    if not torch.cuda.is_available():
        return False
    return bool(torch.cuda.get_device_capability() >= _TF32_MIN_CAPABILITY)


def configure_matmul_precision(*, enabled: bool = True) -> bool:
    """Enable TF32 matmuls when the device supports them. Returns what was set.

    Worth 1.34x on the PPO update on an RTX 4090 (45.8 -> 34.2 ms per
    minibatch), measured with `scripts/measure_throughput.py`'s methodology on
    the real network.

    This lowers matmul mantissa precision from 24 bits to 11, so trained results
    move slightly. That is below every effect size this project can resolve --
    win rate cannot separate differences under ~7pp and `vp_margin` under
    ~10 -- but it does mean a run before this setting and a run after are not
    bit-identical. Environment and reward arithmetic are untouched: they are
    numpy on the CPU and never reach a tensor core.
    """
    if not enabled or not tf32_is_available():
        return False
    torch.set_float32_matmul_precision("high")
    logger.info(
        "TF32 matmuls enabled on {} (sm_{}{})",
        torch.cuda.get_device_name(),
        *torch.cuda.get_device_capability(),
    )
    return True
