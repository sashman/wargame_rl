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

    # Advance when a normal full move cannot reach the objective -- the same
    # "run while far, walk once close" rule the scripted player baselines use.
    #
    # ⚠ A bar the OPPONENT cannot use is the same defect as one the bar cannot
    # use. Until this existed, an advancing agent trained against an opponent
    # walking at Move while it ran at `M + D6`, which flatters the agent at the
    # matchup it is scored on.
    #
    # Decided per GROUP because that is how the env resolves it: one D6 per unit,
    # and `advanced_this_turn` marked for every model of any group that advanced.
    # A per-model decision would forfeit the whole unit's shooting to buy extra
    # distance for one model.
    advance_when_out_of_reach: bool = False

    def select_action(
        self,
        opponent_models: list[WargameModel],
        env: WargameEnv,
        action_mask: np.ndarray | None = None,
    ) -> WargameEnvAction:
        handler = env._opponent_action_handler
        actions: list[int] = []
        obj_locs = np.array([o.location for o in env.objectives])

        alive_models_list = [m for m in opponent_models if m.is_alive]
        if not alive_models_list:
            return WargameEnvAction(actions=[STAY_ACTION] * len(opponent_models))

        alive_locs = np.array([m.location for m in alive_models_list])
        centroid = alive_locs.mean(axis=0)

        w = self._cohesion_weight

        obj_radii = np.array([o.radius_size for o in env.objectives])

        # One decision per unit, taken before any of its models moves.
        speeds = handler.move_speeds
        advancing_groups: set[int] = set()
        if self.advance_when_out_of_reach:
            for group_id in {int(m.group_id) for m in opponent_models if m.is_alive}:
                members = [
                    i
                    for i, m in enumerate(opponent_models)
                    if int(m.group_id) == group_id and m.is_alive
                ]
                reach = float(min(speeds[i] for i in members) if speeds.size else 0.0)
                # Measured from the unit's CENTROID, matching the scripted player
                # baselines: a unit moves as a body, so whether a normal move
                # arrives is decided by where the body is, not by its nearest
                # model. Using the nearest member instead almost never fires --
                # the opponent deploys 3-12" from its objectives at Move 6.
                unit_centre = np.mean(
                    [opponent_models[i].location for i in members], axis=0, dtype=float
                )
                gaps = np.linalg.norm(obj_locs - unit_centre, axis=1) - obj_radii
                if gaps.size and float(np.min(gaps)) > reach:
                    advancing_groups.add(group_id)

        for index, model in enumerate(opponent_models):
            if not model.is_alive:
                actions.append(STAY_ACTION)
                continue

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
            advance = (
                handler.best_advance_toward(
                    dx,
                    dy,
                    advance_roll=float(model.advance_roll),
                    max_step_length=distance_to_boundary,
                    model_idx=index,
                )
                if int(model.group_id) in advancing_groups
                else None
            )
            actions.append(
                advance
                if advance is not None
                else handler.best_action_toward(
                    dx,
                    dy,
                    max_step_length=distance_to_boundary,
                    model_idx=index,
                )
            )

        return WargameEnvAction(actions=actions)


register_policy("scripted_advance_to_objective", ScriptedAdvanceToObjectivePolicy)
