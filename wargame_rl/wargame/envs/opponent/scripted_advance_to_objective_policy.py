"""Scripted policy: opponent models advance toward the nearest objective
while maintaining group cohesion around their centroid."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.envs.opponent.policy import OpponentPolicy
from wargame_rl.wargame.envs.opponent.registry import register_policy
from wargame_rl.wargame.envs.types import WargameEnvAction

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.wargame import WargameEnv
    from wargame_rl.wargame.envs.wargame_model import WargameModel

DEFAULT_COHESION_WEIGHT = 0.3


class ScriptedAdvanceToObjectivePolicy(OpponentPolicy):
    """Move opponent models toward the nearest objective while keeping the
    group together.

    Each step, every model's desired direction is a weighted blend of:
    - the vector toward its nearest objective, and
    - the vector toward the group centroid.

    ``cohesion_weight`` (0–1) controls the balance: 0 = pure objective
    seeking, 1 = pure flocking toward the centroid.
    """

    def __init__(self, env: WargameEnv, **kwargs: object) -> None:
        self._env = env
        self._cohesion_weight = float(
            kwargs.get("cohesion_weight", DEFAULT_COHESION_WEIGHT)  # type: ignore[arg-type]
        )

    def select_action(
        self,
        opponent_models: list[WargameModel],
        env: WargameEnv,
        action_mask: np.ndarray | None = None,
    ) -> WargameEnvAction:
        handler = env._opponent_action_handler
        actions: list[int] = []
        obj_locs = np.array([o.location for o in env.objectives])

        model_locs = np.array([m.location for m in opponent_models])
        centroid = model_locs.mean(axis=0)

        w = self._cohesion_weight

        obj_radii = np.array([o.radius_size for o in env.objectives])

        for model in opponent_models:
            obj_deltas = obj_locs - model.location
            dists = np.linalg.norm(obj_deltas, axis=1)
            nearest_idx = int(np.argmin(dists))

            if dists[nearest_idx] <= obj_radii[nearest_idx]:
                actions.append(STAY_ACTION)
                continue

            to_obj = obj_deltas[nearest_idx]
            to_centroid = centroid - model.location

            obj_norm = dists[nearest_idx]
            centroid_norm = np.linalg.norm(to_centroid)
            obj_dir = to_obj / obj_norm
            centroid_dir = (
                to_centroid / centroid_norm if centroid_norm > 0 else to_centroid
            )

            blended = (1.0 - w) * obj_dir + w * centroid_dir
            dx, dy = float(blended[0]), float(blended[1])
            distance_to_boundary = dists[nearest_idx] - obj_radii[nearest_idx]
            actions.append(
                handler.best_action_toward(dx, dy, max_step_length=distance_to_boundary)
            )

        return WargameEnvAction(actions=actions)


register_policy("scripted_advance_to_objective", ScriptedAdvanceToObjectivePolicy)
