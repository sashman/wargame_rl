"""Pay a model for standing inside its unit's coherent body.

Every other approach to coherency in this repo is a **constraint** -- deploy in
formation, undo a move that breaks it, remove models that cannot regain it. None
of them gives a policy a reason to *stay* in formation, and measurement says it
does not: trained under coherent deployment alone, the agent starts legal and
drifts to a `coherency_rate` of 0.53-0.62 with ~3 of 25 models adrift, because
nothing in the reward ever mentions the rule.

This is the missing lever. Three design rules from
[docs/reward-phases.md](../../../../docs/reward-phases.md) shape it:

**Per-model is not enough; it must be per-model *differentiated*.** A flat
payment to every model in a coherent unit would give the straggler no private
reason to rejoin, exactly as flat `objective_hold` gave the thirteenth model on
a point no reason to leave. So the payment is keyed on *this* model's own
membership of its unit's coherent body: the straggler earns nothing while its
squadmates earn, and the difference is the gradient.

**The agent must be able to observe what the reward keys on.** It can, but only
with `observe_coherency: true`, which puts the spread ratio and the component
fraction on each model's token. Without it the two states this term separates
are identical in the observation, and an unattributable reward is experienced
only as "this pays less". A config setting this calculator without that flag is
rejected at construction rather than trained for eight hours.

**Magnitude matters more than sign.** `group_cohesion` at -0.2 inverted the
baseline ranking and at -0.05 sits in the winning config, so the default here is
deliberately small. It is also *positive*: staying together should pay rather
than breaking up costing, so that the term cannot make manoeuvring toward
objectives a net loss and quietly suppress the whole activity.

**MEASURED, AND IT SATURATES AROUND 0.67. Read this before tuning it.** At
matched epoch 175 on the real-maps scenario: control 0.562, this term at a 0.05
pay gap 0.621, at 0.10 0.659; by epoch 370 the steeper arm reached 0.674.
`vp_margin` was unchanged throughout (~+92), so the term is free -- and it is
also not the lever. Diminishing increments (+0.059 then +0.038 as the gradient
doubled) are the signature of a bonus fighting a stronger term, not of a missing
gradient.

**Why, and it retracts the paragraph that used to sit here.** This calculator
originally claimed it did not conflict with `crowding_exponent` -- that one is
between units and the other within, so five coherent units on five objectives
satisfies both. Measured, that is wrong. 82.4% of adrift models have a *different
nearest objective* from their unit's coherent body, at a median 13.6 board units
from it against a 9.0 spread cap (`scratchpad/defection_check.py`, 1399
observations). They are not drifting, they are **defecting**: pot-splitting sends
the marginal model to the emptiest objective and nothing makes that spread respect
unit boundaries. This term is a +0.05 counterweight to that, which is why it
moves the metric a little and then stops.

So coherency here is a *symptom* of unit-blind objective income. The lever worth
building is **unit-atomic** objective pay, so a lone defector adds nothing its
unit did not already have -- subject to the two standing traps, that an
anti-concentration lever must conserve total income and must stay per-model
differentiated.

**And enforcement dominates it outright.** Warm-starting a trained policy into
`coherency.enforce_move: revert_model` reaches `coherency_rate` **1.000** with
zero models adrift within ten epochs. Training into that constraint from scratch
is what collapses (-75.3 vp_margin, the policy stops moving); from a competent
basin it is close to free. Prefer the constraint. Keep this term for the
measurement it bought, not as the mechanism.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.domain.coherency import evaluate_coherency
from wargame_rl.wargame.envs.domain.entities import alive_mask_for
from wargame_rl.wargame.envs.reward.calculators.base import PerModelRewardCalculator

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.reward.step_context import StepContext
    from wargame_rl.wargame.envs.wargame_model import WargameModel


class UnitCoherencyCalculator(PerModelRewardCalculator):
    """Per-model payment for being inside the unit's largest chain component.

    A model alone in its unit is coherent by definition and is paid, matching
    the rule -- a one-model unit cannot be out of coherency, and taxing the last
    survivor of a squad would price a casualty the model did not cause.
    """

    def __init__(
        self,
        weight: float = 1.0,
        value: float = 0.05,
        straggler_penalty: float = 0.0,
    ) -> None:
        """
        Args:
            weight: Scales the whole term, as for every calculator.
            value: Paid per step to a model inside its unit's coherent body.
                Small on purpose -- around a fifth of `objective_hold`, so the
                term nudges formation without competing with the objective.
            straggler_penalty: Optional extra charge for a model *outside* the
                body, default 0.0. The positive payment alone already
                differentiates; this exists to sharpen the gradient if it does
                not, and is the knob to move before raising `value`, since
                raising `value` inflates total income and changes what the
                policy trades objectives against.
        """
        super().__init__(weight=weight)
        if value < 0.0:
            raise ValueError(f"value must be >= 0, got {value}")
        if straggler_penalty > 0.0:
            raise ValueError(f"straggler_penalty must be <= 0, got {straggler_penalty}")
        self.value = value
        self.straggler_penalty = straggler_penalty
        # One evaluation per step serves every model, keyed on StepContext
        # identity -- the same pattern `group_cohesion` and `objective_hold`
        # use. Its own field, because a key shared between two quantities
        # computed at different points in a step freezes the later one.
        self._cached_ctx: StepContext | None = None
        self._cached_in_body: np.ndarray | None = None

    def reset_episode(self) -> None:
        """Drop the per-step coherency cache."""
        self._cached_ctx = None
        self._cached_in_body = None

    def _in_coherent_body(self, view: BattleView, ctx: StepContext) -> np.ndarray:
        """Per-model: True while the model is inside its unit's coherent body."""
        if ctx is self._cached_ctx and self._cached_in_body is not None:
            return self._cached_in_body

        models = view.player_models
        alive = alive_mask_for(models)
        report = evaluate_coherency(
            positions=np.array([m.location for m in models], dtype=float),
            group_ids=np.array([m.group_id for m in models], dtype=np.intp),
            alive_mask=alive,
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
            # A coherent unit pays every member; a broken one pays only the
            # models still attached to its largest chain component, so the
            # straggler is the one that loses and the body is not punished for
            # a departure it did not make.
            if unit.coherent:
                in_body[unit.member_indices] = True
            else:
                largest = np.bincount(unit.component).argmax()
                in_body[unit.member_indices[unit.component == largest]] = True
        self._cached_ctx = ctx
        self._cached_in_body = in_body
        return in_body

    def calculate(
        self,
        model_idx: int,
        model: WargameModel,
        view: BattleView,
        ctx: StepContext,
    ) -> float:
        """Return this model's coherency payment for the step."""
        if not model.is_alive:
            return 0.0
        if self._in_coherent_body(view, ctx)[model_idx]:
            return float(self.value)
        return float(self.straggler_penalty)
