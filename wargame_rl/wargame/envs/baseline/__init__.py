"""Scripted baseline policies for the player's models.

These are measuring sticks, not opponents: they drive the player's army with
hand-written heuristics so every learned-policy number has a floor and a
reference bar. See `wargame_rl/wargame/envs/opponent/` for the opponent side.
"""

from __future__ import annotations

from wargame_rl.wargame.envs.baseline.policy import BaselinePolicy
from wargame_rl.wargame.envs.baseline.registry import (
    build_baseline_policy,
    get_registry,
    register_baseline,
)

__all__ = [
    "BaselinePolicy",
    "build_baseline_policy",
    "get_registry",
    "register_baseline",
]
