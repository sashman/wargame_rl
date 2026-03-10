from __future__ import annotations

from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.reward.calculators.base import PerModelRewardCalculator

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.reward.step_context import StepContext
    from wargame_rl.wargame.envs.wargame import WargameEnv
    from wargame_rl.wargame.envs.wargame_model import WargameModel


class ClosestObjectiveCalculator(PerModelRewardCalculator):
    """Legacy closest-objective reward to match Reward()."""

    def _get_closest_objective_reward(
        self, previous_model_distance: float, distance_to_closest_objective: float
    ) -> float:
        distance_improvement = float(
            distance_to_closest_objective - previous_model_distance
        )

        if distance_improvement >= 0:
            return -(float(2) * abs(distance_improvement) + float(0.3))

        return float(0)

    def calculate(
        self,
        model_idx: int,
        model: WargameModel,
        env: WargameEnv,
        ctx: StepContext,
    ) -> float:
        cache = ctx.distance_cache
        closest_obj_idx = int(cache.model_obj_norms[model_idx].argmin())
        distance_to_closest = float(
            cache.model_obj_norms_offset[model_idx, closest_obj_idx]
        )

        max_diagonal = float((ctx.board_width**2 + ctx.board_height**2) ** 0.5)
        normalized_distance = distance_to_closest / max_diagonal

        previous = model.previous_closest_objective_distance
        model.set_previous_closest_objective_distance(normalized_distance)

        if previous is None:
            return 0.0

        return self._get_closest_objective_reward(float(previous), normalized_distance)
