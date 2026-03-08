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

        obj_radii = np.array([o.radius_size for o in env.objectives], dtype=float)

        for model in opponent_models:
            obj_deltas = obj_locs - model.location
            dists = np.linalg.norm(obj_deltas, axis=1)
            nearest_idx = int(np.argmin(dists))
            r = obj_radii[nearest_idx]
            obj_loc = obj_locs[nearest_idx]

            # Use same "at objective" rule as game: offset = (model - obj) + (r/2, r/2), inside when norm(offset) <= r
            offset = model.location - obj_loc + (r / 2.0)
            norm_offset = float(np.linalg.norm(offset))
            if norm_offset <= r:
                actions.append(STAY_ACTION)
                continue

            # Move toward capture circle center (obj - r/2) so we land inside the game's capture zone
            capture_center = obj_loc - (r / 2.0)
            to_capture = capture_center - model.location
            to_capture_norm = float(np.linalg.norm(to_capture))
            to_centroid = centroid - model.location
            centroid_norm = np.linalg.norm(to_centroid)
            obj_dir = (
                to_capture / to_capture_norm if to_capture_norm > 0 else to_capture
            )
            centroid_dir = (
                to_centroid / centroid_norm if centroid_norm > 0 else to_centroid
            )

            blended = (1.0 - w) * obj_dir + w * centroid_dir
            dx, dy = float(blended[0]), float(blended[1])
            # Cap movement so we don't overshoot; allow at least 1 cell so we can step inside.
            distance_to_boundary = norm_offset - r
            max_distance = max(distance_to_boundary, 1.0)
            actions.append(
                handler.best_action_toward(dx, dy, max_distance=max_distance)
            )

        return WargameEnvAction(actions=actions)


register_policy("scripted_advance_to_objective", ScriptedAdvanceToObjectivePolicy)
