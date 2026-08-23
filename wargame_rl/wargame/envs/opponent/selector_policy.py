"""Opponent policy: seat any `ActionSelector` on the opponent side.

The env already has one adapter that plays a player-side *baseline* from the
opponent seat (`scripted_baseline`). This is the same idea one level up: it
takes anything with the `ActionSelector` shape — a scripted policy, a network, a
pool snapshot — and hands it a `MirroredEnv` plus an observation built from that
mirror.

**Torch-free on purpose.** A network opponent is
`model/opponent/network_policy.py`, which subclasses this. Putting the torch
half here would make `envs` import `model`, which is both a dependency
inversion and a real import cycle, since `model/net.py` imports `envs.wargame`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeAlias, cast

import numpy as np

from wargame_rl.wargame.envs.env_components.observation_builder import build_observation
from wargame_rl.wargame.envs.opponent.mirror import MirroredEnv
from wargame_rl.wargame.envs.opponent.policy import OpponentPolicy

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.entities import WargameModel
    from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvObservation
    from wargame_rl.wargame.envs.wargame import WargameEnv

# Structurally identical to `envs.baseline.evaluate.ActionSelector`, declared
# here so this module does not depend on the baseline package for a callable
# shape. That alias is what already unifies scripted policies and checkpoints
# everywhere else in the repo.
OpponentSelector: TypeAlias = Callable[
    ["WargameEnvObservation", "WargameEnv"], "WargameEnvAction"
]


class SelectorOpponentPolicy(OpponentPolicy):
    """Drive the opponent army with an `ActionSelector`."""

    def __init__(
        self,
        env: WargameEnv,
        *,
        select: OpponentSelector,
        shoots: bool | None = None,
        label: str = "selector",
    ) -> None:
        self._select = select
        self._label = label
        self._mirror = MirroredEnv(env)
        # Derived from the action space rather than declared, the same
        # "cannot forget" discipline `scripted_baseline` applies to
        # `select_shooting`: if the opponent's handler has a shooting slice then
        # this policy can emit a shot, so the env must pay for the refined mask.
        # Left False, the mask would be phase-and-alive only and every shot
        # would be resolved unchecked.
        self.shoots = (
            env.opponent_action_handler.shooting_slice is not None
            if shoots is None
            else shoots
        )

    @property
    def label(self) -> str:
        """What to call this opponent in a table or a recording."""
        return self._label

    @property
    def mirror(self) -> WargameEnv:
        """The side-swapped view this policy plays through."""
        return cast("WargameEnv", self._mirror)

    def select_action(
        self,
        opponent_models: list[WargameModel],
        env: WargameEnv,
        action_mask: np.ndarray | None = None,
    ) -> WargameEnvAction:
        """Build the opponent's own observation and ask the selector.

        `action_registry=None` is deliberate and is a throughput decision, not
        an oversight. `_apply_opponent_action` has already built a fully
        rules-legal mask — range, line of sight, engagement — and passes it in;
        letting `build_observation` build its own would run
        `compute_unit_shooting_masks`, and with it the line-of-sight pass, a
        second time in every shooting phase. So the mask is spliced in rather
        than recomputed.
        """
        observation = build_observation(self._mirror, action_registry=None)  # type: ignore[arg-type]
        if action_mask is not None:
            observation = dataclasses.replace(observation, action_mask=action_mask)
        return self._select(observation, cast("WargameEnv", self._mirror))
