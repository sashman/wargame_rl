from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.reward.calculators.base import GlobalRewardCalculator

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.reward.step_context import StepContext


class ObjectiveFlipBonusCalculator(GlobalRewardCalculator):
    """Big sparse reward for objective state upgrades.

    Rewarded transitions:
    - neutral -> player-controlled
    - opponent-controlled -> contested
    - contested -> player-controlled

    Control state per objective uses in-range counts:
    - player-controlled: player_count >= opponent_count + 1
    - opponent-controlled: opponent_count >= player_count + 1
    - contested: both present and margin is < 1
    - neutral: no one present
    """

    def __init__(
        self,
        weight: float = 1.0,
        bonus_capture_first: float = 5.0,
        bonus_flip_to_contested: float = 3.0,
        bonus_contested_to_controlled: float = 5.0,
        unique_per_objective: bool = True,
    ) -> None:
        super().__init__(weight=weight)
        self.bonus_capture_first = bonus_capture_first
        self.bonus_flip_to_contested = bonus_flip_to_contested
        self.bonus_contested_to_controlled = bonus_contested_to_controlled
        self.unique_per_objective = unique_per_objective
        self._previous_states: list[str] | None = None
        self._rewarded_objectives: set[int] = set()
        self._last_turn: int | None = None
        self._last_step_key: tuple[int, int] | None = None

    @staticmethod
    def _state_label(player_count: int, opponent_count: int) -> str:
        if player_count <= 0 and opponent_count <= 0:
            return "neutral"
        if player_count >= opponent_count + 1:
            return "player"
        if opponent_count >= player_count + 1:
            return "opponent"
        return "contested"

    @staticmethod
    def _counts_by_objective(
        player_in_range: np.ndarray,
        opponent_in_range: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        player_counts = np.sum(player_in_range, axis=0)
        opponent_counts = np.sum(opponent_in_range, axis=0)
        return player_counts, opponent_counts

    def _current_states(self, view: BattleView, ctx: StepContext) -> list[str]:
        cache = ctx.distance_cache
        player_in_range = cache.model_obj_norms_offset <= cache.obj_radii
        if view.opponent_models:
            from wargame_rl.wargame.envs.domain.entities import alive_mask_for
            from wargame_rl.wargame.envs.env_components.distance_cache import (
                compute_distances,
            )

            opp_alive = alive_mask_for(view.opponent_models)
            opponent_cache = compute_distances(
                view.opponent_models,
                view.objectives,
                alive_mask=opp_alive,
            )
            opponent_in_range = (
                opponent_cache.model_obj_norms_offset <= opponent_cache.obj_radii
            )
        else:
            n_obj = len(view.objectives)
            opponent_in_range = np.zeros((0, n_obj), dtype=bool)

        player_counts, opponent_counts = self._counts_by_objective(
            player_in_range, opponent_in_range
        )
        return [
            self._state_label(int(player_counts[i]), int(opponent_counts[i]))
            for i in range(len(view.objectives))
        ]

    def calculate(self, view: BattleView, ctx: StepContext) -> float:
        step_key = (ctx.current_turn, id(ctx.distance_cache))
        # New episode heuristic (supports consecutive 1-step episodes).
        if self._last_turn is not None and ctx.current_turn < self._last_turn:
            self._previous_states = None
            self._rewarded_objectives.clear()
        elif (
            self._last_turn is not None
            and ctx.current_turn == 1
            and self._last_turn == 1
            and self._last_step_key is not None
            and self._last_step_key != step_key
        ):
            self._previous_states = None
            self._rewarded_objectives.clear()
        self._last_turn = ctx.current_turn
        self._last_step_key = step_key

        current_states = self._current_states(view, ctx)
        if self._previous_states is None or len(self._previous_states) != len(
            current_states
        ):
            self._previous_states = current_states
            return 0.0

        bonus = 0.0
        for obj_idx, (prev, cur) in enumerate(
            zip(self._previous_states, current_states)
        ):
            if self.unique_per_objective and obj_idx in self._rewarded_objectives:
                continue

            gained = 0.0
            if prev == "neutral" and cur == "player":
                gained = self.bonus_capture_first
            elif prev == "opponent" and cur == "contested":
                gained = self.bonus_flip_to_contested
            elif prev == "contested" and cur == "player":
                gained = self.bonus_contested_to_controlled

            if gained > 0.0:
                bonus += gained
                if self.unique_per_objective:
                    self._rewarded_objectives.add(obj_idx)

        self._previous_states = current_states
        return bonus
