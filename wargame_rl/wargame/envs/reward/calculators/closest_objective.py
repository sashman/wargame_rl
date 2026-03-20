from __future__ import annotations

from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.reward.calculators.base import PerModelRewardCalculator

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.reward.step_context import StepContext
    from wargame_rl.wargame.envs.wargame_model import WargameModel


class ClosestObjectiveCalculator(PerModelRewardCalculator):
    """Legacy closest-objective reward to match Reward().

    Optionally adds a bonus when a model achieves a new best (lowest) distance
    to its closest objective, scaled by the improvement in normalized distance.
    """

    def __init__(
        self,
        weight: float = 1.0,
        best_distance_bonus_scale: float | None = None,
    ) -> None:
        super().__init__(weight=weight)
        self.best_distance_bonus_scale = (
            0.0 if best_distance_bonus_scale is None else best_distance_bonus_scale
        )

    @staticmethod
    def _normalized_distance(
        ctx: StepContext, distance_to_closest_objective: float
    ) -> float:
        max_diagonal = float((ctx.board_width**2 + ctx.board_height**2) ** 0.5)
        return distance_to_closest_objective / max_diagonal

    @staticmethod
    def _penalty_for_non_improvement(
        previous_model_distance: float, current_distance: float
    ) -> float:
        """Penalty when the model fails to get closer (or stays the same)."""
        distance_delta = float(current_distance - previous_model_distance)

        if distance_delta >= 0:
            return -(float(2) * abs(distance_delta) + float(0.3))

        return 0.0

    def _best_distance_bonus(
        self, previous_best: float | None, current_distance: float
    ) -> float:
        """Bonus when a new best (lowest) distance is achieved."""
        if previous_best is None or current_distance >= previous_best:
            return 0.0
        if self.best_distance_bonus_scale == 0.0:
            return 0.0
        improvement = float(previous_best - current_distance)
        return self.best_distance_bonus_scale * (improvement**3)

    def calculate(
        self,
        model_idx: int,
        model: WargameModel,
        view: BattleView,
        ctx: StepContext,
    ) -> float:
        cache = ctx.distance_cache
        closest_obj_idx = int(cache.model_obj_norms[model_idx].argmin())
        distance_to_closest = float(
            cache.model_obj_norms_offset[model_idx, closest_obj_idx]
        )

        normalized_distance = self._normalized_distance(ctx, distance_to_closest)

        previous = model.previous_closest_objective_distance
        model.set_previous_closest_objective_distance(normalized_distance)

        best_prev = model.best_closest_objective_distance
        if best_prev is None or normalized_distance < best_prev:
            model.set_best_closest_objective_distance(normalized_distance)

        bonus = self._best_distance_bonus(best_prev, normalized_distance)

        if previous is None:
            return bonus

        base_penalty = self._penalty_for_non_improvement(
            float(previous), normalized_distance
        )
        return base_penalty + bonus
