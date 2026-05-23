from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.domain.entities import alive_mask_for
from wargame_rl.wargame.envs.env_components.distance_cache import compute_distances
from wargame_rl.wargame.envs.reward.calculators.base import PerModelRewardCalculator

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.env_components.distance_cache import DistanceCache
    from wargame_rl.wargame.envs.reward.step_context import StepContext
    from wargame_rl.wargame.envs.wargame_model import WargameModel


class ClosestObjectiveV2Calculator(PerModelRewardCalculator):
    """Closest-objective shaping that targets only net-VP-impactful objectives.

    A target objective is considered net-positive for the player when this model's
    arrival can improve objective control:
    - neutral -> player-controlled
    - opponent-controlled -> contested

    Objectives already controlled/contested by a friendly model are ignored.
    If no net-positive target exists, this calculator returns 0 for the model.
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
        self._last_breakdown: dict[int, dict[str, float]] = {}
        self._target_obj_idx: dict[int, int] = {}
        self._previous_target_distance: dict[int, float] = {}
        self._best_target_distance: dict[int, float] = {}
        self._cached_step_key: tuple[int, int] | None = None
        self._cached_player_any: np.ndarray | None = None
        self._cached_opponent_any: np.ndarray | None = None

    @staticmethod
    def _normalized_distance(ctx: StepContext, distance_to_objective: float) -> float:
        max_diagonal = float((ctx.board_width**2 + ctx.board_height**2) ** 0.5)
        return distance_to_objective / max_diagonal

    @staticmethod
    def _penalty_for_non_improvement(
        previous_model_distance: float, current_distance: float
    ) -> float:
        distance_delta = float(current_distance - previous_model_distance)
        if distance_delta >= 0:
            return -(float(2) * abs(distance_delta) + float(0.3))
        return 0.0

    def _best_distance_bonus(
        self, previous_best: float | None, current_distance: float
    ) -> float:
        if previous_best is None or current_distance >= previous_best:
            return 0.0
        if self.best_distance_bonus_scale == 0.0:
            return 0.0
        improvement = float(previous_best - current_distance)
        return self.best_distance_bonus_scale * (improvement**3)

    def _clear_model_state(self, model_idx: int) -> None:
        self._target_obj_idx.pop(model_idx, None)
        self._previous_target_distance.pop(model_idx, None)
        self._best_target_distance.pop(model_idx, None)

    def _objective_presence_masks(
        self,
        view: BattleView,
        ctx: StepContext,
    ) -> tuple[np.ndarray, np.ndarray]:
        step_key = (ctx.current_turn, id(ctx.distance_cache))
        if (
            self._cached_step_key == step_key
            and self._cached_player_any is not None
            and self._cached_opponent_any is not None
        ):
            return self._cached_player_any, self._cached_opponent_any

        cache = ctx.distance_cache
        player_any = np.any(cache.model_obj_norms_offset <= cache.obj_radii, axis=0)

        n_obj = len(view.objectives)
        if view.opponent_models:
            opp_alive = alive_mask_for(view.opponent_models)
            opponent_cache = compute_distances(
                view.opponent_models,
                view.objectives,
                alive_mask=opp_alive,
            )
            opponent_any = np.any(
                opponent_cache.model_obj_norms_offset <= opponent_cache.obj_radii,
                axis=0,
            )
        else:
            opponent_any = np.zeros(n_obj, dtype=bool)

        self._cached_step_key = step_key
        self._cached_player_any = player_any
        self._cached_opponent_any = opponent_any
        return player_any, opponent_any

    def _choose_target_objective(
        self,
        model_idx: int,
        cache: DistanceCache,
        player_any: np.ndarray,
    ) -> int | None:
        model_distances = cache.model_obj_norms_offset[model_idx]
        obj_radii = cache.obj_radii
        model_outside = model_distances > obj_radii

        # If player already has presence, moving this model there cannot improve
        # control state (already player-controlled or already contested).
        net_positive = np.logical_not(player_any)
        candidates = model_outside & net_positive
        if not np.any(candidates):
            return None

        candidate_indices = np.flatnonzero(candidates)
        closest_idx = int(
            candidate_indices[np.argmin(model_distances[candidate_indices])]
        )
        return closest_idx

    def calculate(
        self,
        model_idx: int,
        model: WargameModel,
        view: BattleView,
        ctx: StepContext,
    ) -> float:
        cache = ctx.distance_cache
        player_any, _ = self._objective_presence_masks(view, ctx)
        target_obj_idx = self._choose_target_objective(model_idx, cache, player_any)

        if target_obj_idx is None:
            self._clear_model_state(model_idx)
            self._last_breakdown[model_idx] = {
                "target_obj_idx": -1.0,
                "target_switched": 0.0,
                "distance_delta": 0.0,
                "base_penalty": 0.0,
                "best_distance_bonus": 0.0,
            }
            return 0.0

        distance_to_target = float(
            cache.model_obj_norms_offset[model_idx, target_obj_idx]
        )
        normalized_distance = self._normalized_distance(ctx, distance_to_target)

        previous_target = self._target_obj_idx.get(model_idx)
        if previous_target != target_obj_idx:
            self._target_obj_idx[model_idx] = target_obj_idx
            self._previous_target_distance[model_idx] = normalized_distance
            self._best_target_distance[model_idx] = normalized_distance
            self._last_breakdown[model_idx] = {
                "target_obj_idx": float(target_obj_idx),
                "target_switched": 1.0,
                "distance_delta": 0.0,
                "base_penalty": 0.0,
                "best_distance_bonus": 0.0,
            }
            return 0.0

        previous = self._previous_target_distance.get(model_idx)
        self._previous_target_distance[model_idx] = normalized_distance

        best_prev = self._best_target_distance.get(model_idx)
        if best_prev is None or normalized_distance < best_prev:
            self._best_target_distance[model_idx] = normalized_distance

        bonus = self._best_distance_bonus(best_prev, normalized_distance)

        if previous is None:
            self._last_breakdown[model_idx] = {
                "target_obj_idx": float(target_obj_idx),
                "target_switched": 0.0,
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
            "target_obj_idx": float(target_obj_idx),
            "target_switched": 0.0,
            "distance_delta": distance_delta,
            "base_penalty": base_penalty,
            "best_distance_bonus": bonus,
        }
        return base_penalty + bonus

    def get_last_breakdown(self, model_idx: int) -> dict[str, float]:
        return self._last_breakdown.get(model_idx, {})
