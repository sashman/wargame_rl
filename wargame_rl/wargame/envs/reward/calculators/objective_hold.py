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
# 0.0 = an objective pays every occupant in full, the historical behaviour.
DEFAULT_CROWDING_EXPONENT = 0.0


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

    ``crowding_exponent`` divides an objective's value by ``occupants ** a``, so
    the point pays a *pot* rather than a per-head wage. This is the answer to a
    measured failure and not a guess: at 1000 epochs the agent ends with 15.8
    models alive, of which **12.9 stand on an objective defended by 0.25
    opponents** while the second objective is lost 4.2 to 2.7. Roughly 14 models
    are surplus to control, and re-allocating them would take ``objectives_held``
    from 1.42 to 2.06 -- past the 1.64 of ``squad_march_shoot``. See
    ``scripts/measure_objective_split.py``.

    The flat default cannot express that: the thirteenth model on a point is paid
    exactly like the first, so no model ever has a private reason to leave.

    **Two earlier levers at this same defect failed**, and the property they
    lacked is the one to preserve in anything that replaces this. An overstack
    penalty and a surplus discount (both since removed) each *lowered total
    objective income*, so the policy experienced them as "objectives pay less"
    and did fewer of them -- occupancy 0.925 -> 0.520 and 0.784 -> 0.284. At
    ``a = 1`` the pot is instead conserved: total pay across a point's occupants
    is its value however many stand there, so spreading onto a second point
    strictly *raises* income. The gradient points at the behaviour rather than
    away from objectives.

    Do not read the win as "objectives should pay more". The confound was
    controlled: the same weight with ``a = 0`` scores **-40.4 vp_margin**
    against this term's +28.4, piling 20 of 21 survivors onto one point. At
    fixed weight the exponent alone is worth 68 vp_margin.

    ``a`` also auto-regulates magnitude -- dividing by ~10 occupants keeps the
    effective pay near the flat term's -- and no experiment separates that from
    the crowding price. Raising the weight without the exponent is what breaks.
    """

    def __init__(
        self,
        weight: float = 1.0,
        player_value: float = DEFAULT_PLAYER_VALUE,
        contested_value: float = DEFAULT_CONTESTED_VALUE,
        opponent_value: float = DEFAULT_OPPONENT_VALUE,
        crowding_exponent: float = DEFAULT_CROWDING_EXPONENT,
    ) -> None:
        super().__init__(weight=weight)
        self.player_value = player_value
        self.contested_value = contested_value
        self.opponent_value = opponent_value
        if crowding_exponent < 0.0:
            raise ValueError(
                f"crowding_exponent must be >= 0, got {crowding_exponent}: a "
                "negative exponent would pay *more* for crowding."
            )
        self.crowding_exponent = crowding_exponent
        self._cached_ctx: StepContext | None = None
        self._cached_values: np.ndarray | None = None
        # Occupancy changes every step and is read by every model, so it gets
        # the same treat-the-context-as-the-key caching as the values do -- and
        # its own key, because `_objective_values` runs first and stamps
        # `_cached_ctx`, so sharing that key would freeze occupancy at step one
        # and price the whole episode at the opening crowd.
        self._cached_occupancy_ctx: StepContext | None = None
        self._cached_occupancy: np.ndarray | None = None

    def reset_episode(self) -> None:
        """Drop the per-step control-state cache."""
        self._cached_ctx = None
        self._cached_values = None
        self._cached_occupancy_ctx = None
        self._cached_occupancy = None

    def _player_occupancy(self, ctx: StepContext) -> np.ndarray:
        """``(n_objectives,)`` count of live player models inside each disc."""
        if ctx is self._cached_occupancy_ctx and self._cached_occupancy is not None:
            return self._cached_occupancy
        cache = ctx.distance_cache
        occupancy = (cache.model_obj_norms_offset <= cache.obj_radii).sum(axis=0)
        self._cached_occupancy = np.atleast_1d(occupancy).astype(np.float64)
        self._cached_occupancy_ctx = ctx
        return self._cached_occupancy

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

        scaled = values
        if self.crowding_exponent != 0.0:
            occupancy = np.maximum(self._player_occupancy(ctx), 1.0)
            scaled = scaled / occupancy**self.crowding_exponent

        return float(np.max(scaled[inside]))
