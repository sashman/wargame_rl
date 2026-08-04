"""Baseline: uniform random legal actions — the floor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.baseline.policy import BaselinePolicy
from wargame_rl.wargame.envs.baseline.registry import register_baseline
from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.envs.types import WargameEnvAction

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.entities import WargameModel
    from wargame_rl.wargame.envs.wargame import WargameEnv


class RandomBaselinePolicy(BaselinePolicy):
    """Pick a uniformly random movement action for every alive model.

    The floor of the scale. A policy that cannot beat this has learned
    nothing — which is the comparison that was missing when a trained network
    was reported at 17% with no reference at either end.
    """

    def __init__(self, seed: int | None = None) -> None:
        # Seeded explicitly rather than via global numpy state so baseline
        # measurements are reproducible run to run.
        self._rng = np.random.default_rng(seed)

    def select_movement(
        self, models: list[WargameModel], env: WargameEnv
    ) -> WargameEnvAction:
        """Return a uniformly random movement action per alive model."""
        n_move_actions = env.player_action_handler.n_move_actions
        actions: list[int] = []
        for model in models:
            if not model.is_alive:
                actions.append(STAY_ACTION)
                continue
            actions.append(int(self._rng.integers(0, n_move_actions + 1)))
        return WargameEnvAction(actions=actions)


register_baseline("random", RandomBaselinePolicy)
