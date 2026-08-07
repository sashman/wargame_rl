"""Per-model reward for occupying an objective, scaled by who controls it."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.domain.entities import alive_mask_for
from wargame_rl.wargame.envs.env_components.distance_cache import (
    compute_distances,
    objective_states_from_norms_offset,
)
from wargame_rl.wargame.envs.reward.calculators.base import PerModelRewardCalculator

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.reward.step_context import StepContext
    from wargame_rl.wargame.envs.wargame_model import WargameModel

DEFAULT_PLAYER_VALUE = 1.0
DEFAULT_CONTESTED_VALUE = 0.5
DEFAULT_OPPONENT_VALUE = 0.25
# 1.0 = surplus models are paid exactly like the ones holding the point, which
# is the historical behaviour every existing config and checkpoint assumes.
DEFAULT_SURPLUS_VALUE = 1.0


class ObjectiveHoldCalculator(PerModelRewardCalculator):
    """Per-model reward for standing on an objective, scaled by control state.

    This is the only calculator that pays a model for what it is doing while
    *stationary*. Every other per-model term goes silent once models arrive:
    ``closest_objective``'s progress is a potential that exhausts on arrival and
    is exactly zero on shooting steps, and ``group_cohesion`` returns a hard 0
    inside its limit. Without this, all models share an identical reward for
    roughly three quarters of an episode, which defeats per-model credit
    assignment.

    It pays for *controlling*, not merely standing, so it strictly dominates the
    global ``models_at_objectives`` — which every competent scripted baseline
    saturates at 1.0, making that term unable to tell a weak policy from a
    strong one. Scaling by control state also supplies a gradient across the
    ``neutral -> contested -> player`` region that ``objective_coverage`` and
    ``vp_gain`` are both flat across, since control is a strict count
    comparison.
    """

    def __init__(
        self,
        weight: float = 1.0,
        player_value: float = DEFAULT_PLAYER_VALUE,
        contested_value: float = DEFAULT_CONTESTED_VALUE,
        opponent_value: float = DEFAULT_OPPONENT_VALUE,
        surplus_value: float = DEFAULT_SURPLUS_VALUE,
    ) -> None:
        super().__init__(weight=weight)
        self.player_value = player_value
        self.contested_value = contested_value
        self.opponent_value = opponent_value
        self.surplus_value = surplus_value
        self._cached_ctx: StepContext | None = None
        self._cached_values: np.ndarray | None = None
        # Its own cache key: `_objective_values` runs first and stamps
        # `_cached_ctx`, so sharing that key would freeze the quota after
        # step one and pay the same models for the whole episode.
        self._cached_quota_ctx: StepContext | None = None
        self._cached_within_quota: np.ndarray | None = None

    def reset_episode(self) -> None:
        """Drop the per-step control-state cache."""
        self._cached_ctx = None
        self._cached_values = None
        self._cached_quota_ctx = None
        self._cached_within_quota = None

    def _within_quota(self, view: BattleView, ctx: StepContext) -> np.ndarray:
        """``(n_models, n_objectives)`` bool: is this model *needed* to hold that point?

        Control is a strict count comparison, so an objective needs
        ``opponent_count + 1`` models and every model beyond that changes
        nothing about who scores it. The models nearest the centre are counted
        as the holders; the rest are surplus.

        This is what makes "take a second objective" pay more than "add a
        sixteenth model to the first". Without it the calculator is indifferent
        between 15 models on one point and 8/7 across two -- both pay
        ``15 x player_value`` -- while VP is worth double for the second.
        """
        cache = ctx.distance_cache
        inside = cache.model_obj_norms_offset <= cache.obj_radii

        if view.opponent_models:
            opponent_alive = alive_mask_for(view.opponent_models)
            opponent_norms = compute_distances(
                view.opponent_models, view.objectives, alive_mask=opponent_alive
            ).model_obj_norms_offset
            opponent_counts = (opponent_norms <= cache.obj_radii).sum(axis=0)
        else:
            opponent_counts = np.zeros(len(view.objectives), dtype=int)

        within = np.zeros_like(inside, dtype=bool)
        for obj_idx in range(inside.shape[1]):
            occupants = np.flatnonzero(inside[:, obj_idx])
            if occupants.size == 0:
                continue
            quota = int(opponent_counts[obj_idx]) + 1
            # Nearest to the centre are the holders. Any deterministic rule
            # works -- what matters is that the same models keep the full value
            # from step to step, or the reward would flicker between them.
            nearest = occupants[
                np.argsort(cache.model_obj_norms_offset[occupants, obj_idx])
            ]
            within[nearest[:quota], obj_idx] = True
        return within

    def _value_for_state(self, state: str) -> float:
        if state == "player":
            return self.player_value
        if state == "contested":
            return self.contested_value
        if state == "opponent":
            return self.opponent_value
        return 0.0  # neutral: occupied by nobody, so holding it is worth nothing

    def _objective_values(self, view: BattleView, ctx: StepContext) -> np.ndarray:
        """Per-objective value, computed once per step and reused across models.

        `calculate` is invoked once per model, and deriving control needs the
        opponent's distances, so without this the opponent cache would be
        rebuilt 25 times a step. A fresh `StepContext` is built every step and
        held by the env, so identity comparison is a safe cache key.
        """
        if ctx is self._cached_ctx and self._cached_values is not None:
            return self._cached_values

        n_objectives = len(view.objectives)
        if view.opponent_models:
            opponent_alive = alive_mask_for(view.opponent_models)
            opponent_norms = compute_distances(
                view.opponent_models, view.objectives, alive_mask=opponent_alive
            ).model_obj_norms_offset
        else:
            opponent_norms = np.zeros((0, n_objectives), dtype=np.float64)

        states = objective_states_from_norms_offset(
            ctx.distance_cache.model_obj_norms_offset,
            opponent_norms,
            ctx.distance_cache.obj_radii,
        )
        values = np.array([self._value_for_state(s) for s in states], dtype=np.float64)
        self._cached_ctx = ctx
        self._cached_values = values
        return values

    def calculate(
        self,
        model_idx: int,
        model: WargameModel,
        view: BattleView,
        ctx: StepContext,
    ) -> float:
        """Return the control-scaled value of the objective this model occupies.

        A model inside overlapping objectives is credited with the best of them.
        """
        if not view.objectives:
            return 0.0
        cache = ctx.distance_cache
        inside = cache.model_obj_norms_offset[model_idx] <= cache.obj_radii
        if not bool(np.any(inside)):
            return 0.0
        values = self._objective_values(view, ctx)

        if self.surplus_value == 1.0:
            return float(np.max(values[inside]))

        if ctx is not self._cached_quota_ctx or self._cached_within_quota is None:
            self._cached_within_quota = self._within_quota(view, ctx)
            self._cached_quota_ctx = ctx
        within = self._cached_within_quota[model_idx]

        scaled = np.where(within, values, values * self.surplus_value)
        return float(np.max(scaled[inside]))
