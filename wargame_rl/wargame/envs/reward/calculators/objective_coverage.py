"""Dense reward for the fraction of objectives the player controls."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.env_components.distance_cache import (
    objective_ownership_from_norms_offset,
)
from wargame_rl.wargame.envs.reward.calculators.base import GlobalRewardCalculator

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.reward.step_context import StepContext


class ObjectiveCoverageCalculator(GlobalRewardCalculator):
    """Dense reward = fraction of objectives the player controls.

    Returns (number of player-controlled objectives) / (number of objectives),
    using the same OC/count control rule as VP scoring. Because it is paid every
    step, it gives a smooth incentive to **spread** models and hold *multiple
    distinct* objectives rather than over-stacking one — complementing
    ``closest_objective_v2``'s per-group de-stacking. Returns unweighted; the
    phase manager applies the configured weight.
    """

    def calculate(self, view: BattleView, ctx: StepContext) -> float:
        cache = ctx.distance_cache
        n_obj = len(view.objectives)
        if n_obj == 0:
            return 0.0
        if view.opponent_models:
            opponent_norms = ctx.opponent_distances(view).model_obj_norms_offset
        else:
            opponent_norms = np.zeros((0, n_obj), dtype=np.float64)
        player_controls, _ = objective_ownership_from_norms_offset(
            cache.model_obj_norms_offset, opponent_norms, cache.obj_radii
        )
        return float(np.sum(player_controls)) / float(n_obj)
