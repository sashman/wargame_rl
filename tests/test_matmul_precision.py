"""TF32 must stay off unless a caller asks for it.

Measured 2026-08-09 on `configs/golden/25v25_shooting_opponent.yaml`: TF32 costs
~8.5 vp_margin (two seeds, n=100, epoch 1000 on both sides) for 17.8% of an
epoch. The `--no-tf32` control reproduced the pre-TF32 run bit-identically, so
the setting is the whole of the difference.

The failure this guards against is silent and expensive: flipping the default
back costs 8.5 vp with nothing raising, and win rate would not resolve it
(0.705 -> 0.65, inside the ~7pp limit). It is caught only by scoring two full
1000-epoch runs against each other -- seven GPU-hours to notice.
"""

from __future__ import annotations

import torch

from wargame_rl.wargame.model.common.performance import (
    configure_matmul_precision,
    tf32_is_available,
)


def test_tf32_is_off_by_default() -> None:
    """Calling with no argument must not enable TF32, on any device."""
    assert configure_matmul_precision() is False


def test_opting_in_enables_tf32_where_the_device_allows_it() -> None:
    """`enabled=True` is honoured, so the escape hatch actually works.

    Restores the global afterwards: `set_float32_matmul_precision` is
    process-wide, and leaking 'high' would silently change every test that
    builds a network after this one.
    """
    previous = torch.get_float32_matmul_precision()
    try:
        assert configure_matmul_precision(enabled=True) is tf32_is_available()
    finally:
        torch.set_float32_matmul_precision(previous)


def test_the_default_leaves_the_global_untouched() -> None:
    """The default path must not write the global at all, not merely write 'highest'."""
    previous = torch.get_float32_matmul_precision()
    try:
        configure_matmul_precision()
        assert torch.get_float32_matmul_precision() == previous
    finally:
        torch.set_float32_matmul_precision(previous)
