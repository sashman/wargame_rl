"""Per-model reward for occupying an objective, scaled by who controls it."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.domain.coherency import evaluate_coherency
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


def _states_from_counts(player: np.ndarray, opponent: np.ndarray) -> list[str]:
    """Control state per objective from the two counts, under VP's own rule.

    Strictly more models controls it; equal and non-zero is contested; empty is
    neutral. Kept beside the calculator rather than reusing
    `objective_states_from_norms_offset`, which derives the counts from
    distances -- the marginal value needs to ask "what if this model were not
    here", which is a question about counts, not positions.
    """
    states: list[str] = []
    for p, q in zip(player, opponent):
        if p == 0 and q == 0:
            states.append("neutral")
        elif p > q:
            states.append("player")
        elif q > p:
            states.append("opponent")
        else:
            states.append("contested")
    return states


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
        marginal_weight: float = 0.0,
        require_coherent: bool = False,
    ) -> None:
        super().__init__(weight=weight)
        if not 0.0 <= marginal_weight <= 1.0:
            raise ValueError(
                f"marginal_weight must be in [0, 1], got {marginal_weight}"
            )
        self.marginal_weight = marginal_weight
        self.require_coherent = require_coherent
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
        self._cached_body: np.ndarray | None = None
        self._cached_body_ctx: StepContext | None = None
        self._cached_opp: np.ndarray | None = None
        self._cached_opp_ctx: StepContext | None = None
        self._cached_marginal: np.ndarray | None = None
        self._cached_marginal_ctx: StepContext | None = None
        self._cached_occupancy_ctx: StepContext | None = None
        self._cached_occupancy: np.ndarray | None = None

    def reset_episode(self) -> None:
        """Drop the per-step control-state cache."""
        self._cached_ctx = None
        self._cached_values = None
        self._cached_occupancy_ctx = None
        self._cached_marginal = None
        self._cached_marginal_ctx = None
        self._cached_opp = None
        self._cached_opp_ctx = None
        self._cached_body = None
        self._cached_body_ctx = None
        self._cached_occupancy = None

    def _in_coherent_body(self, view: BattleView, ctx: StepContext) -> np.ndarray:
        """Per-model: True while the model is inside its unit's coherent body.

        The same predicate `domain/coherency.py` serves to the metric and to
        enforcement, so the reward and the rule cannot drift apart.

        **Why this gate is the shape it is.** Under the occupancy form a unit's
        income depends only on how many objectives it touches, never on how far
        anyone walked: measured per step at weight 1.25, five models together
        earn the unit 1.25, a legal 3+2 across two nearby objectives earns 2.50,
        and a 4+1 with one model detached far away earns **the same 2.50**. The
        reward cannot tell the legal spread from the illegal one, and full
        scatter pays 6.25 -- five times staying together. That is the gradient
        the adrift models are following: 82.4% of them are walking to a
        different objective from their unit's body.

        Gating on the coherency predicate makes the reward agree with the rule.
        The legal 3+2 is untouched, because every model is in the body and the
        9" spread cap is what makes that spread legal in the first place. The
        4+1 collapses to 1.25, because the detached model earns nothing.
        """
        if ctx is self._cached_body_ctx and self._cached_body is not None:
            return self._cached_body

        models = view.player_models
        report = evaluate_coherency(
            positions=np.array([m.location for m in models], dtype=float),
            group_ids=np.array([m.group_id for m in models], dtype=np.intp),
            alive_mask=alive_mask_for(models),
            base_radii=np.array([m.base_radius for m in models], dtype=float),
            nearest_distance=view.rules_quantities.scale.to_units(
                view.config.coherency.nearest_distance
            ),
            furthest_distance=view.rules_quantities.scale.to_units(
                view.config.coherency.furthest_distance
            ),
        )
        in_body = np.zeros(len(models), dtype=bool)
        for unit in report.units:
            if unit.coherent:
                in_body[unit.member_indices] = True
            else:
                largest = np.bincount(unit.component).argmax()
                in_body[unit.member_indices[unit.component == largest]] = True
        self._cached_body = in_body
        self._cached_body_ctx = ctx
        return in_body

    def _player_occupancy(self, ctx: StepContext) -> np.ndarray:
        """``(n_objectives,)`` count of live player models inside each disc."""
        if ctx is self._cached_occupancy_ctx and self._cached_occupancy is not None:
            return self._cached_occupancy
        cache = ctx.distance_cache
        occupancy = (cache.model_obj_norms_offset <= cache.obj_radii).sum(axis=0)
        self._cached_occupancy = np.atleast_1d(occupancy).astype(np.float64)
        self._cached_occupancy_ctx = ctx
        return self._cached_occupancy

    def _opponent_occupancy(self, view: BattleView, ctx: StepContext) -> np.ndarray:
        """``(n_objectives,)`` count of live opponent models inside each disc."""
        if ctx is self._cached_opp_ctx and self._cached_opp is not None:
            return self._cached_opp
        n_objectives = len(view.objectives)
        if view.opponent_models:
            opponent_norms = compute_distances(
                view.opponent_models,
                view.objectives,
                alive_mask=alive_mask_for(view.opponent_models),
            ).model_obj_norms_offset
            counts = (opponent_norms <= ctx.distance_cache.obj_radii).sum(axis=0)
        else:
            counts = np.zeros(n_objectives)
        self._cached_opp = np.atleast_1d(counts).astype(np.float64)
        self._cached_opp_ctx = ctx
        return self._cached_opp

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
        # A model outside its unit's coherent body is in a state the rules do
        # not allow -- `03-moving.md` says the move cannot be made. An illegal
        # position should not be paid, so it earns nothing here.
        if self.require_coherent and not self._in_coherent_body(view, ctx)[model_idx]:
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

        if self.marginal_weight > 0.0:
            marginal = self._marginal_values(view, ctx)
            scaled = (
                1.0 - self.marginal_weight
            ) * scaled + self.marginal_weight * marginal

        return float(np.max(scaled[inside]))

    def _marginal_values(self, view: BattleView, ctx: StepContext) -> np.ndarray:
        """Per-objective value of *one* player model's presence, `V(p) - V(p-1)`.

        A **difference reward**: what does this model's standing here actually
        change? Computed on the control counts, which the agent can see with
        `observe_objective_control`.

        It prices the two defects the occupancy form cannot:

        - **Futile defection.** One model against three defenders leaves the
          objective the opponent's either way, so its marginal value is 0.
          Under the occupancy form it collects the full opponent-state value,
          which is why 82.4% of adrift models are walking to a *different*
          objective from their unit's body -- they are paid to.
        - **Over-stacking.** The sixth model on a point held 5-to-2 changes
          nothing, so it earns nothing. `crowding_exponent` approximates this by
          dividing; this derives it.

        The model that *flips* control earns the whole swing, which is the
        behaviour worth paying for.

        Note what it does to a **safely held** point: every model there has a
        marginal value of 0, so the term stops paying for holding ground nobody
        contests. That is a real reduction in total income, and this repo has
        twice measured that a lever which destroys income is experienced as
        "this activity pays less" -- hence `marginal_weight` blends rather than
        replaces, and defaults to 0.0, which is byte-identical to before it
        existed. `vp_gain` and `objective_coverage` still pay for the hold.
        """
        if ctx is self._cached_marginal_ctx and self._cached_marginal is not None:
            return self._cached_marginal

        player = self._player_occupancy(ctx)
        opponent = self._opponent_occupancy(view, ctx)
        here = np.array(
            [self._value_for_state(s) for s in _states_from_counts(player, opponent)],
            dtype=np.float64,
        )
        without = np.array(
            [
                self._value_for_state(s)
                for s in _states_from_counts(np.maximum(player - 1.0, 0.0), opponent)
            ],
            dtype=np.float64,
        )
        self._cached_marginal = here - without
        self._cached_marginal_ctx = ctx
        return self._cached_marginal
