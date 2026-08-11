"""Baseline: march squads onto objectives, and shoot when a target is valid."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.baseline.registry import register_baseline
from wargame_rl.wargame.envs.baseline.scripted_squad_march import (
    ScriptedSquadMarchPolicy,
)
from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.envs.types import WargameEnvAction

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.entities import WargameModel
    from wargame_rl.wargame.envs.wargame import WargameEnv


class ScriptedSquadMarchShootPolicy(ScriptedSquadMarchPolicy):
    """`squad_march`, plus firing at the nearest valid target each shooting phase.

    The reference bar for a policy class the agent is actually in. Every other
    baseline is movement-only, so their ~0.78 win rate is the ceiling of a
    policy that never fires — while the agent gets 20 shooting decisions per
    episode against an opponent that cannot shoot back. Calibrating the final
    curriculum gate against a movement-only score would aim below what a
    competent shooter should reach.

    Target choice is nearest-first rather than best-expected-damage. With a
    homogeneous army every target has identical expected damage, so the two
    agree; nearest is simpler and stays correct if stats diverge only slightly.
    """

    def select_shooting(
        self,
        models: list[WargameModel],
        env: WargameEnv,
        action_mask: np.ndarray | None = None,
    ) -> WargameEnvAction:
        """Fire each model at its nearest valid target, or hold if it has none."""
        shooting_slice = env.player_action_handler.shooting_slice
        actions = [STAY_ACTION] * len(models)
        if shooting_slice is None or action_mask is None:
            return WargameEnvAction(actions=actions)

        opponents = env.opponent_models
        if not opponents:
            return WargameEnvAction(actions=actions)
        opponent_locations = np.array([m.location for m in opponents], dtype=float)
        opponent_groups = np.array([m.group_id for m in opponents], dtype=int)

        for index, model in enumerate(models):
            if not model.is_alive:
                continue
            # The env mask already encodes range, line of sight, target-alive
            # and engagement-range validity, so honouring it is what keeps this
            # baseline playing by the same rules as the learned policy.
            valid = np.flatnonzero(
                action_mask[index, shooting_slice.start : shooting_slice.end]
            )
            if valid.size == 0:
                continue
            # `valid` indexes enemy UNITS, not models: a weapon names a unit.
            # Nearest unit means the unit whose closest model is closest, which
            # is also the distance the range check was made against.
            here = np.asarray(model.location, dtype=float)
            model_distances = np.linalg.norm(opponent_locations - here, axis=1)
            unit_distances = [
                model_distances[opponent_groups == unit].min()
                if (opponent_groups == unit).any()
                else np.inf
                for unit in valid
            ]
            target = int(valid[int(np.argmin(unit_distances))])
            actions[index] = shooting_slice.start + target

        return WargameEnvAction(actions=actions)


register_baseline("squad_march_shoot", ScriptedSquadMarchShootPolicy)
