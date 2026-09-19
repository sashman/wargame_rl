"""Checkpoints for the per-model driver: plain tensors, no pickled env.

A Lightning checkpoint here pickles the whole env as a hyper-parameter, which
is why every reader of one needs `weights_only=False`. This format holds the
state dict, the two configs as plain dicts, the head sizes the network was
built with, and the run's provenance -- loadable with `weights_only=True`,
and rebuildable without the env that trained it.

Periodic, not exit-hooked: SIGKILL is the prescribed way to stop a trainer
and it triggers no handler, so `last.pt` is written every interval and is at
most one interval stale.

Two ways back in. A **resume** needs the optimizer moments, the sampling
generator and the driver's running counters, which travel as additive keys
(`optimizer_state`, `generator_state`, `declarations_seen`,
`approx_kl_cumulative`) so every checkpoint written before they existed still
loads for play; `load_training_state` refuses one of those by name. A **warm
start** needs the weights alone -- the size-independent path the curriculum
climbs on -- and goes through `load_checkpoint`, which refuses a displacement
head of the wrong width.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from wargame_rl.wargame.envs.per_model.types import FACADE_TAG
from wargame_rl.wargame.model.per_model.config import SetNetworkConfig
from wargame_rl.wargame.model.per_model.net import SetNetwork
from wargame_rl.wargame.model.per_model.ppo import PerModelPPOConfig

LAST_CHECKPOINT = "last.pt"


def periodic_checkpoint_name(rounds: int) -> str:
    return f"pm-{rounds:08d}.pt"


@dataclass(frozen=True)
class TrainingState:
    """What a resume needs beyond the weights: the optimizer's moments, the
    sampling generator mid-stream, and the driver's running counters."""

    optimizer_state: dict[str, Any]
    generator_state: torch.Tensor
    declarations_seen: int
    approx_kl_cumulative: float
    # The KL anchor's adapted coefficient (#332); 0.0 when the run has none.
    kl_ref_coef: float = 0.0


def save_checkpoint(
    path: Path,
    network: SetNetwork,
    *,
    ppo_config: PerModelPPOConfig,
    env_config: dict[str, Any],
    rounds: int,
    seed: int | None,
    revision: str,
    training_state: TrainingState | None = None,
) -> None:
    """Write the checkpoint atomically (`.tmp` then replace).

    `training_state` adds the resume keys; without it the checkpoint plays
    and warm-starts but cannot be resumed, which `load_training_state` says.
    """
    payload: dict[str, Any] = {
        "facade": FACADE_TAG,
        "state_dict": {k: v.detach().cpu() for k, v in network.state_dict().items()},
        "network_config": network.config.model_dump(),
        "head_sizes": {"n_displacements": network.n_displacements},
        # JSON mode: a `Credit` enum member is not a global the weights-only
        # loader admits; the string round-trips through the model.
        "ppo_config": ppo_config.model_dump(mode="json"),
        "env_config": env_config,
        "rounds": int(rounds),
        "seed": seed,
        "revision": revision,
    }
    if training_state is not None:
        payload["optimizer_state"] = _to_cpu(training_state.optimizer_state)
        payload["generator_state"] = training_state.generator_state.clone()
        payload["declarations_seen"] = int(training_state.declarations_seen)
        payload["approx_kl_cumulative"] = float(training_state.approx_kl_cumulative)
        payload["kl_ref_coef"] = float(training_state.kl_ref_coef)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".tmp")
    torch.save(payload, partial)
    os.replace(partial, path)


def _to_cpu(value: Any) -> Any:
    """Optimizer state, tensors moved to the CPU, containers rebuilt."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


@dataclass(frozen=True)
class LoadedCheckpoint:
    """A checkpoint read back: the network rebuilt, plus what it carried."""

    network: SetNetwork
    ppo_config: PerModelPPOConfig
    env_config: dict[str, Any]
    rounds: int
    seed: int | None
    revision: str


def load_checkpoint(
    path: Path, *, expected_n_displacements: int | None = None
) -> LoadedCheckpoint:
    """Rebuild the network from `path`; refuses a head-size mismatch by name."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("facade") != FACADE_TAG:
        raise ValueError(
            f"{path} is not a per-model checkpoint (facade={payload.get('facade')!r})"
        )
    head_sizes = payload["head_sizes"]
    n_displacements = int(head_sizes["n_displacements"])
    if (
        expected_n_displacements is not None
        and expected_n_displacements != n_displacements
    ):
        raise ValueError(
            f"{path} was trained with a displacement head of {n_displacements} "
            f"columns; this scenario's action encoding has "
            f"{expected_n_displacements}"
        )
    network = SetNetwork(
        SetNetworkConfig(**payload["network_config"]),
        n_displacements=n_displacements,
    )
    network.load_state_dict(payload["state_dict"])
    return LoadedCheckpoint(
        network=network,
        ppo_config=PerModelPPOConfig(**payload["ppo_config"]),
        env_config=dict(payload["env_config"]),
        rounds=int(payload["rounds"]),
        seed=payload.get("seed"),
        revision=str(payload.get("revision", "")),
    )


def load_training_state(path: Path) -> TrainingState:
    """The resume keys of `path`; refuses a checkpoint written without them."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    missing = [
        key
        for key in ("optimizer_state", "generator_state", "declarations_seen")
        if key not in payload
    ]
    if missing:
        raise ValueError(
            f"{path} carries no training state ({', '.join(missing)} missing): "
            "it was written before resume existed, or by a tool that saves "
            "weights only. Warm-start from it instead (`--warm-start-from`)."
        )
    return TrainingState(
        optimizer_state=dict(payload["optimizer_state"]),
        generator_state=payload["generator_state"],
        declarations_seen=int(payload["declarations_seen"]),
        approx_kl_cumulative=float(payload.get("approx_kl_cumulative", 0.0)),
        kl_ref_coef=float(payload.get("kl_ref_coef", 0.0)),
    )


__all__ = [
    "LAST_CHECKPOINT",
    "LoadedCheckpoint",
    "TrainingState",
    "load_checkpoint",
    "load_training_state",
    "periodic_checkpoint_name",
    "save_checkpoint",
]
