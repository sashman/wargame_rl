"""Scripted policy: each opponent model advances toward the nearest objective."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.opponent.policy import OpponentPolicy
from wargame_rl.wargame.envs.opponent.registry import register_policy
from wargame_rl.wargame.envs.types import WargameEnvAction

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.wargame import WargameEnv
    from wargame_rl.wargame.envs.wargame_model import WargameModel


class ScriptedAdvanceToObjectivePolicy(OpponentPolicy):
    """Move each opponent model toward its nearest objective at max speed."""

    def __init__(self, env: WargameEnv, **kwargs: object) -> None:
        self._env = env

    def select_action(
        self,
        opponent_models: list[WargameModel],
        env: WargameEnv,
        action_mask: np.ndarray | None = None,
    ) -> WargameEnvAction:
        handler = env._opponent_action_handler
        actions: list[int] = []
        obj_locs = np.array([o.location for o in env.objectives])

        for model in opponent_models:
            deltas = obj_locs - model.location
            dists = np.linalg.norm(deltas, axis=1)
            nearest_idx = int(np.argmin(dists))
            dx, dy = float(deltas[nearest_idx, 0]), float(deltas[nearest_idx, 1])
            actions.append(handler.best_action_toward(dx, dy))

        return WargameEnvAction(actions=actions)


register_policy("scripted_advance_to_objective", ScriptedAdvanceToObjectivePolicy)
