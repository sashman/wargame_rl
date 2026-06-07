from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.domain.entities import alive_mask_for
from wargame_rl.wargame.envs.env_components.distance_cache import (
    compute_distances,
    objective_states_from_norms_offset,
)
from wargame_rl.wargame.envs.reward.calculators.base import GlobalRewardCalculator

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.reward.step_context import StepContext


class ObjectiveFlipBonusCalculator(GlobalRewardCalculator):
    """Symmetric objective-control shaping (a control-state potential).

    Each objective's control state (same OC/count rule as VP scoring) maps to a
    scalar value from the player's POV; the per-step reward is the change in that
    value summed over objectives. Gains use the configured bonuses; losses
    subtract the mirror value scaled by ``loss_penalty_scale``. At
    ``loss_penalty_scale == 1.0`` the term is a pure (farming-proof) potential.

    Gain transitions reproduce the configured magnitudes exactly:
    - neutral   -> player    == bonus_capture_first
    - opponent  -> contested == bonus_flip_to_contested
    - contested -> player    == bonus_contested_to_controlled
    """

    def __init__(
        self,
        weight: float = 1.0,
        bonus_capture_first: float = 5.0,
        bonus_flip_to_contested: float = 3.0,
        bonus_contested_to_controlled: float = 5.0,
        loss_penalty_scale: float = 1.0,
    ) -> None:
        super().__init__(weight=weight)
        self.bonus_capture_first = bonus_capture_first
        self.bonus_flip_to_contested = bonus_flip_to_contested
        self.bonus_contested_to_controlled = bonus_contested_to_controlled
        self.loss_penalty_scale = loss_penalty_scale
        self._previous_states: list[str] | None = None

    def reset_episode(self) -> None:
        """Clear per-episode state (called by the env on reset)."""
        self._previous_states = None

    def _state_value(self, state: str) -> float:
        if state == "opponent":
            return 0.0
        if state == "contested":
            return self.bonus_flip_to_contested
        if state == "player":
            return self.bonus_flip_to_contested + self.bonus_contested_to_controlled
        # neutral
        return (
            self.bonus_flip_to_contested
            + self.bonus_contested_to_controlled
            - self.bonus_capture_first
        )

    def _current_states(self, view: BattleView, ctx: StepContext) -> list[str]:
        cache = ctx.distance_cache
        n_obj = len(view.objectives)
        if view.opponent_models:
            opp_alive = alive_mask_for(view.opponent_models)
            opponent_cache = compute_distances(
                view.opponent_models, view.objectives, alive_mask=opp_alive
            )
            opponent_norms = opponent_cache.model_obj_norms_offset
        else:
            opponent_norms = np.zeros((0, n_obj), dtype=np.float64)
        return objective_states_from_norms_offset(
            cache.model_obj_norms_offset, opponent_norms, cache.obj_radii
        )

    def calculate(self, view: BattleView, ctx: StepContext) -> float:
        current_states = self._current_states(view, ctx)
        if self._previous_states is None or len(self._previous_states) != len(
            current_states
        ):
            self._previous_states = current_states
            return 0.0

        bonus = 0.0
        for prev, cur in zip(self._previous_states, current_states):
            delta = self._state_value(cur) - self._state_value(prev)
            if delta < 0.0:
                delta *= self.loss_penalty_scale
            bonus += delta

        self._previous_states = current_states
        return bonus
