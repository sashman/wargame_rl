"""Who is being rated.

An entrant is a name plus a way to build a playable selector for a given env.
It is a **factory** rather than a selector because a network entrant must be
sized from the env it will play in, and a scripted entrant holds env state --
neither is safe to carry across configs, and each leg is its own config.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.baseline.evaluate import ActionSelector
    from wargame_rl.wargame.envs.wargame import WargameEnv


@dataclass(frozen=True, slots=True)
class Entrant:
    """One competitor in a rating table."""

    name: str
    build: Callable[[WargameEnv], ActionSelector]
    kind: Literal["baseline", "checkpoint"]
    source: str | None = None
    decode_topk: int = 1
    # True when the selector draws from the policy rather than taking its
    # argmax. Self-play rollouts do; every scoring path in this repo does not.
    # It is recorded because a sampled player and a greedy one are two different
    # players, and a ledger holding both under one name would rank the decode.
    sampled: bool = False
    # Which checkpoint this descended from. Two seeds off one warm start are not
    # two independent samples -- checkpoints differing by +0.067 in unit
    # coherency produced descendants differing by +0.19 after 300 epochs -- so a
    # pool that does not record lineage cannot tell a real effect from an
    # inherited one.
    parent: str | None = None
