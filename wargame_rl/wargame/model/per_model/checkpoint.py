"""Checkpoints for the per-model driver: plain tensors, no pickled env.

A Lightning checkpoint here pickles the whole env as a hyper-parameter, which
is why every reader of one needs `weights_only=False`. This format holds the
state dict, the two configs as plain dicts, the head sizes the network was
built with, and the run's provenance -- loadable with `weights_only=True`,
and rebuildable without the env that trained it.

Periodic, not exit-hooked: SIGKILL is the prescribed way to stop a trainer
and it triggers no handler, so `last.pt` is written every interval and is at
most one interval stale.
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


def save_checkpoint(
    path: Path,
    network: SetNetwork,
    *,
    ppo_config: PerModelPPOConfig,
    env_config: dict[str, Any],
    rounds: int,
    seed: int | None,
    revision: str,
) -> None:
    """Write the checkpoint atomically (`.tmp` then replace)."""
    payload = {
        "facade": FACADE_TAG,
        "state_dict": {k: v.detach().cpu() for k, v in network.state_dict().items()},
        "network_config": network.config.model_dump(),
        "head_sizes": {"n_displacements": network.n_displacements},
        "ppo_config": ppo_config.model_dump(),
        "env_config": env_config,
        "rounds": int(rounds),
        "seed": seed,
        "revision": revision,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".tmp")
    torch.save(payload, partial)
    os.replace(partial, path)


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


__all__ = [
    "LAST_CHECKPOINT",
    "LoadedCheckpoint",
    "load_checkpoint",
    "periodic_checkpoint_name",
    "save_checkpoint",
]
