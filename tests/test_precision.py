"""Precision settings: TF32 gating, and the float32 guard on the PPO heads."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch

from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.performance import (
    configure_matmul_precision,
    tf32_is_available,
)
from wargame_rl.wargame.model.ppo.ppo import PPO_Transformer

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


@pytest.fixture(autouse=True)
def restore_matmul_precision() -> Iterator[None]:
    """`set_float32_matmul_precision` is process-wide, so put it back."""
    previous = torch.get_float32_matmul_precision()
    yield
    torch.set_float32_matmul_precision(previous)


def test_disabled_never_touches_the_global_setting() -> None:
    torch.set_float32_matmul_precision("highest")

    assert configure_matmul_precision(enabled=False) is False
    assert torch.get_float32_matmul_precision() == "highest"


def test_reports_what_it_actually_set() -> None:
    """The return value tracks the device, so a caller cannot assume TF32."""
    torch.set_float32_matmul_precision("highest")

    enabled = configure_matmul_precision(enabled=True)

    assert enabled is tf32_is_available()
    assert torch.get_float32_matmul_precision() == ("high" if enabled else "highest")


@requires_cuda
def test_tf32_availability_follows_compute_capability() -> None:
    """TF32 needs Ampere. Below sm_80 torch accepts the setting and ignores it."""
    assert tf32_is_available() == (torch.cuda.get_device_capability() >= (8, 0))


@requires_cuda
def test_heads_return_float32_under_bf16_autocast(env: WargameEnv) -> None:
    """PPO's importance ratio must not be computed in 8 mantissa bits.

    `exp(new_log_prob - old_log_prob)` has to resolve per-model log-prob changes
    of ~0.007 nats; bf16 cannot represent that near 1.0, so the ratio would
    collapse to exactly 1 and the surrogate objective would carry no gradient.
    """
    from wargame_rl.wargame.model.common.observation import observation_to_tensor

    net = PPO_Transformer.from_env(env).to("cuda")
    observation, _ = env.reset(seed=0)
    tensors = [t.to("cuda") for t in observation_to_tensor(observation)]

    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, values = net(tensors)

    assert logits.dtype is torch.float32
    assert values.dtype is torch.float32


def test_bf16_cannot_resolve_a_log_prob_change_at_this_magnitude() -> None:
    """The guard is load-bearing, at the magnitude these log-probs actually have.

    A 122-way categorical sits near log p = -4.8, where bf16's 8 mantissa bits
    space values 0.0156 apart. The 0.007 nats PPO's ratio must resolve is under
    half of that, so it does not survive the round trip: it collapses to zero for
    most base values and is inflated to one or two whole steps for the rest.
    Never anything near 0.007 -- which is the point. It is not that bf16 is
    merely noisy here; it cannot represent the signal at all.
    """
    base = torch.linspace(-6.0, -3.0, 2001, dtype=torch.float32)

    coarse_delta = (base + 0.007).to(torch.bfloat16).float() - base.to(
        torch.bfloat16
    ).float()

    assert torch.unique(coarse_delta).tolist() == [0.0, 0.015625, 0.03125]
    assert (coarse_delta == 0.0).float().mean() > 0.5
    assert ((coarse_delta - 0.007).abs() < 0.0007).sum() == 0
