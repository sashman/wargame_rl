from __future__ import annotations

from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.reward.calculators.base import PerModelRewardCalculator

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.reward.step_context import StepContext
    from wargame_rl.wargame.envs.wargame_model import WargameModel


class ClosestObjectiveCalculator(PerModelRewardCalculator):
    """Legacy closest-objective reward to match Reward().

    Two shaping modes against the nearest objective (distance normalized by the
    board diagonal):

    - ``progress_scale > 0`` (recommended): a smooth linear potential pull,
      ``reward = progress_scale * distance_closed`` (positive when closing,
      negative when receding). Speed-to-objective is then driven by the env
      discount (gamma) and the remaining-fraction terminal bonus, not by a
      convex distance term.
    - legacy (default): 0 when getting closer, a flat penalty when distance
      stays the same or grows, plus an optional cubic best-distance bonus
      (``best_distance_bonus_scale``).
    """

    def __init__(
        self,
        weight: float = 1.0,
        best_distance_bonus_scale: float | None = None,
        progress_scale: float = 0.0,
    ) -> None:
        super().__init__(weight=weight)
        self.best_distance_bonus_scale = (
            0.0 if best_distance_bonus_scale is None else best_distance_bonus_scale
        )
        # When > 0, use a smooth linear potential pull toward the nearest
        # objective instead of the legacy flat penalty + cubic best-distance
        # bonus. A convex (cubic) term is a poor proxy for "arrive fast" (it only
        # fires on a new best distance and is wildly scale-sensitive); discounting
        # and the remaining-fraction terminal bonus carry the speed incentive.
        self.progress_scale = progress_scale
        self._last_breakdown: dict[int, dict[str, float]] = {}

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

        if self.progress_scale > 0.0:
            # Linear potential pull toward the nearest objective: positive when
            # closing distance, negative when receding. Smooth and dense.
            distance_delta = (
                0.0 if previous is None else float(normalized_distance - previous)
            )
            progress = (
                0.0 if previous is None else self.progress_scale * (-distance_delta)
            )
            self._last_breakdown[model_idx] = {
                "distance_delta": distance_delta,
                "base_penalty": 0.0,
                "best_distance_bonus": 0.0,
                "progress": progress,
            }
            return progress

        bonus = self._best_distance_bonus(best_prev, normalized_distance)

        if previous is None:
            self._last_breakdown[model_idx] = {
                "distance_delta": 0.0,
                "base_penalty": 0.0,
                "best_distance_bonus": bonus,
            }
            return bonus

        distance_delta = float(normalized_distance - previous)
        base_penalty = self._penalty_for_non_improvement(
            float(previous), normalized_distance
        )
        self._last_breakdown[model_idx] = {
            "distance_delta": distance_delta,
            "base_penalty": base_penalty,
            "best_distance_bonus": bonus,
        }
        return base_penalty + bonus

    def get_last_breakdown(self, model_idx: int) -> dict[str, float]:
        return self._last_breakdown.get(model_idx, {})
