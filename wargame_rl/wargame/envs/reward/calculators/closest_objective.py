from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.reward.calculators.base import PerModelRewardCalculator

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.reward.step_context import StepContext
    from wargame_rl.wargame.envs.wargame import WargameEnv
    from wargame_rl.wargame.envs.wargame_model import WargameModel


class ClosestObjectiveCalculator(PerModelRewardCalculator):
    """Reward based on whether a model moved closer to its nearest objective.

    Distances are normalised by the board diagonal so that the improvement
    signal is comparable across different board sizes.
    """

    REWARD_AT_OBJECTIVE = 1.0
    REWARD_CLOSER = 0.5
    PENALTY_NO_CHANGE = -0.05
    PENALTY_FARTHER = -0.5

    def calculate(
        self,
        model_idx: int,
        model: WargameModel,
        env: WargameEnv,
        ctx: StepContext,
    ) -> float:
        cache = ctx.distance_cache
        max_diagonal = float(np.sqrt(ctx.board_width**2 + ctx.board_height**2))

        closest_obj_idx = int(cache.model_obj_norms[model_idx].argmin())
        distance_to_closest = float(
            cache.model_obj_norms_offset[model_idx, closest_obj_idx]
        )
        normalized_distance = distance_to_closest / max_diagonal
        objective_radius_normalized = (
            float(cache.obj_radii[closest_obj_idx]) / max_diagonal
        )

        previous = model.previous_closest_objective_distance
        model.set_previous_closest_objective_distance(normalized_distance)

        if previous is None:
            return 0.0

        previous = float(previous)

        if normalized_distance <= objective_radius_normalized:
            # # One-time bonus when first reaching an objective; no farming while camping.
            # return (
            #     self.REWARD_AT_OBJECTIVE
            #     if previous > objective_radius_normalized
            #     else 0.0
            # )
            return self.REWARD_AT_OBJECTIVE

        improvement = previous - normalized_distance
        if improvement == 0:
            return self.PENALTY_NO_CHANGE
        if improvement > 0:
            return self.REWARD_CLOSER * improvement
        if improvement < 0:
            return self.PENALTY_FARTHER * improvement

        return 0.0
