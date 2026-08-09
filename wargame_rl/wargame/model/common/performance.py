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


def configure_matmul_precision(*, enabled: bool = False) -> bool:
    """Enable TF32 matmuls when the device supports them. Returns what was set.

    **Off by default: TF32 costs ~8.5 vp_margin on 25v25.** Measured 2026-08-09
    on `configs/golden/25v25_shooting_opponent.yaml`, two seeds, n=100 on
    identical layouts, scored at epoch 1000 on both sides:

    | seed | TF32 off | TF32 on | delta |
    |------|----------|---------|-------|
    | 1    | +30.8    | +21.2   | -9.6  |
    | 2    | +27.4    | +19.9   | -7.5  |

    Against a `squad_march_shoot` bar of +17.0, that is the difference between
    beating the bar by 12.1 and beating it by 3.6. The `--no-tf32` control
    reproduced the pre-TF32 run **bit-identically** (222/222 tensors, max abs
    diff 0.0, both seeds), so TF32 is the whole of the difference and nothing
    else in the intervening code touched training.

    This docstring previously claimed the effect was "below every effect size
    this project can resolve". That was reasoned from the mantissa drop (24 bits
    to 11) and the speedup benchmark, never measured against a trained result.
    It is wrong: 8.5 vp is at the resolution limit, not beneath it, and it is
    signed the same way on both seeds. Note win rate *would* have missed it
    (0.705 -> 0.65, inside the ~7pp win-rate limit) -- read `vp_margin`.

    The speed it buys is also smaller than the update benchmark implies: 1.34x
    on the PPO update alone (45.8 -> 34.2 ms per minibatch) is only **17.8%** of
    an epoch end to end (12.95 -> 10.99 s/epoch, same measurement), because the
    update is one part of an epoch. 18% wall clock for 8.5 vp is a bad trade,
    which is why this defaults off rather than merely being documented.

    Pass `--tf32` to opt in when throughput matters more than the result --
    a smoke test, a profiling run, a throughput measurement. Runs differing in
    this setting are not comparable: reproducing a run requires matching it as
    well as the seed. Environment and reward arithmetic are untouched either
    way: they are numpy on the CPU and never reach a tensor core.
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
