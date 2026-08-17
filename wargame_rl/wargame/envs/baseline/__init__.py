"""Scripted baseline policies for the player's models.

These are measuring sticks first: they drive the player's army with hand-written
heuristics so every learned-policy number has a floor and a reference bar.

They can also be played on the *opponent* side, through the `scripted_baseline`
opponent policy (`envs/opponent/scripted_baseline_policy.py`), which hands a
baseline a side-swapped view of the env rather than duplicating it. Nothing here
needs to know about that — but a new baseline that reads a side-specific env
attribute does have to be mirrored, and
`tests/test_scripted_baseline_opponent.py` fails if one is not.
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
