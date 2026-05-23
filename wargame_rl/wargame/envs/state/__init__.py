"""Canonical game-state snapshot: serialisable representation of env state."""

from __future__ import annotations

from wargame_rl.wargame.envs.state.narrator import StepNarrator
from wargame_rl.wargame.envs.state.snapshot import (
    GameStateSnapshot,
    JsonEncoder,
    SnapshotEncoder,
    build_snapshot,
    describe_action,
    validate_snapshot,
)

__all__ = [
    "GameStateSnapshot",
    "JsonEncoder",
    "SnapshotEncoder",
    "StepNarrator",
    "build_snapshot",
    "describe_action",
    "validate_snapshot",
]
