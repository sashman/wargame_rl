"""Pay a charging model for how close it came to a legal charge.

**Why this exists.** `ActionHandler._enforce_charge` is all-or-nothing: a unit
that ends coherent and engaged with exactly one enemy unit keeps its move, and
one that misses by a hair is put back where it started. So a near-miss and a
wild miss produce *identical* feedback -- none -- and the only signal a policy
ever gets about charging is the one it almost never observes.

Measured, that is exactly what happens. Three behaviour clones of a charging
teacher reproduced its shooting at **0.99** and its movement at **0.63** while
echoing **0.8-2.4%** of its actual charge orders: they had not failed to
coordinate, they had failed to *declare*. Forcing the declaration and letting
each model pick its own charge rung landed **0.30-0.39** of charges against the
teacher's 0.85 and an untrained network's 0.00-0.07, so the coordination is
about halfway there for free. **The bottleneck is proposing a rare
all-or-nothing action, not executing a coordinated one**, and a gradient toward
the feasible set is the direct answer to that.

**The three standing checks, answered.**

*Per-model and differentiated.* Each model is paid on **its own** gap to the
nearest enemy, so within one charging unit the model that closed and the model
that lagged earn differently -- which is the gradient. A flat per-unit payment
would give the lagging model no private reason to keep up, exactly as flat
`objective_hold` gave the thirteenth model on a point no reason to leave.

*Does the wanted behaviour pay more in total?* Yes, and strictly. The term is a
potential in `[0, value]` that is **maximal exactly when the model is engaged**,
so a charge that stands pays every member the most this term can pay. Nothing
here pays more for missing than for hitting.

*Can the agent observe what it keys on?* Yes, and this had to be MADE true:
enemy positions are on the opponent tokens, `charge_roll` reached the
observation on 2026-08-25, and `declared_charge` -- the state this term gates
on -- reached it in the same change. ⚠ Before that a correctly-gated term would
have keyed on something the network could not perceive, which is the cheapest
standing check in CLAUDE.md and the one that has burned ~10 GPU-hours twice.

⚠ **Gated on the unit having DECLARED a charge**, and that gate is the whole
difference between this and a travel reward. It reads `model.declared_charge`,
which the leader sets in the command phase and which is cleared at the start of
its side's next turn -- so it is live for exactly the phases this term pays in. `closest_objective_v2` is the
four-times-refuted "walk toward the thing" term, and a 2026-08-11 teleport audit
priced walking a squad at defended ground at **−29.4 of its own income and 1.69
of 5 models**. This pays only inside the charge phase, only to a unit that chose
to charge, and only for the distance that choice was about.

⚠ **UNTRAINED, and it carries a real risk worth stating before anyone spends a
GPU on it.** The 2x2 says the mechanic is worth about zero: both sides walking
scores −2.58 and both charging −3.42, and all of charging's apparent value is
the *unilateral* cell. So this term could successfully teach a behaviour that
does not pay. **Reject rule, pre-registered:** reject if `vp_margin` falls
against a `charge_progress`-free control on 2 of 3 seeds, *or* if charges
declared per episode rises while `held` falls -- that combination is the term
buying charges with ground, which is the trade this project has been caught by
three times.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.domain.entities import alive_mask_for
from wargame_rl.wargame.envs.reward.calculators.base import PerModelRewardCalculator
from wargame_rl.wargame.envs.types.game_timing import BattlePhase

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.reward.step_context import StepContext
    from wargame_rl.wargame.envs.wargame_model import WargameModel


class ChargeProgressCalculator(PerModelRewardCalculator):
    """Per-model payment for closing on the enemy during a declared charge."""

    def __init__(self, weight: float = 1.0, value: float = 0.05) -> None:
        """
        Args:
            weight: Scales the whole term, as for every calculator.
            value: The most a model can earn in a step, paid when it is engaged.
                Small on purpose and the same default as `unit_coherency`: this
                nudges a rare action into being proposed, and a term large
                enough to make charging attractive on its own would be pricing
                the mechanic rather than revealing it.
        """
        super().__init__(weight=weight)
        if value < 0.0:
            raise ValueError(f"value must be >= 0, got {value}")
        self.value = value
        # One scan per step serves every model, keyed on StepContext identity.
        # Its own field: a key shared between two quantities computed at
        # different points in a step freezes the later one.
        self._cached_ctx: StepContext | None = None
        self._cached_progress: np.ndarray | None = None

    def reset_episode(self) -> None:
        """Drop the per-step cache."""
        self._cached_ctx = None
        self._cached_progress = None

    def _progress(self, view: BattleView, ctx: StepContext) -> np.ndarray:
        """Per-model progress toward contact, in `[0, 1]`, zero off the charge."""
        if ctx is self._cached_ctx and self._cached_progress is not None:
            return self._cached_progress

        models = view.player_models
        progress = np.zeros(len(models), dtype=float)
        phase = view.game_clock_state.phase
        enemies = [m for m in view.opponent_models if m.is_alive]
        if (
            not view.config.melee.enabled
            or phase is not BattlePhase.charge
            or not enemies
        ):
            self._cached_ctx = ctx
            self._cached_progress = progress
            return progress

        quantities = view.rules_quantities
        # Contact is base to base, exactly as the engagement predicate measures
        # it -- a centre-to-centre reading would make "engaged" unreachable and
        # cap the term below its own maximum.
        contact = quantities.engagement_range + 2.0 * quantities.base_radius
        reach = quantities.scale.to_units(view.config.melee.charge_range)
        span = max(reach - contact, 1e-6)

        alive = alive_mask_for(models)
        positions = np.array([m.location for m in models], dtype=float)
        enemy_positions = np.array([m.location for m in enemies], dtype=float)
        gaps = np.linalg.norm(
            positions[:, np.newaxis, :] - enemy_positions[np.newaxis, :, :], axis=2
        ).min(axis=1)
        # ⚠ Only a unit that DECLARED. `charged_this_turn` is set by the referee
        # on a charge that STOOD, so it cannot be the gate -- it would pay only
        # the successes and give the near-misses nothing, which is the very
        # discontinuity this term exists to remove. `fell_back_this_turn` and
        # `advanced_this_turn` mark units the rules forbid to charge at all.
        for index, model in enumerate(models):
            if not alive[index] or model.advanced_this_turn:
                continue
            if model.fell_back_this_turn:
                continue
            # ⚠ THE GATE, and it was ABSENT until 2026-08-25. This loop used to
            # test `charge_roll <= 0.0` and call that the declaration check --
            # but `_roll_charge_dice` rolls 2D6 for every unit of the active
            # side unconditionally, and 2D6 is never <= 0, so it excluded
            # nothing. The term paid every alive model for closing on the
            # nearest enemy: `squad_march_take`, which declares ZERO charges,
            # earned 5.713 per episode with 0.0% of it reaching a declared unit,
            # against the charging script's 4.196. It paid the wrong behaviour
            # 36% MORE, which makes it the four-times-refuted "walk toward the
            # thing" term aimed at ENEMIES -- the direction the 2026-08-11
            # teleport audit priced at -29.4 of the committing squad's own
            # income.
            if not model.declared_charge:
                continue
            # Kept BESIDE the declaration gate, not replaced by it. A roll of
            # zero means this side has not rolled yet, and a declaration
            # standing without a roll is an inconsistent state rather than a
            # chargeable one.
            if float(model.charge_roll) <= 0.0:
                continue
            progress[index] = float(np.clip((reach - gaps[index]) / span, 0.0, 1.0))

        self._cached_ctx = ctx
        self._cached_progress = progress
        return progress

    def calculate(
        self,
        model_idx: int,
        model: WargameModel,
        view: BattleView,
        ctx: StepContext,
    ) -> float:
        """Return this model's charge-progress payment for the step."""
        if not model.is_alive:
            return 0.0
        return float(self.value) * float(self._progress(view, ctx)[model_idx])
