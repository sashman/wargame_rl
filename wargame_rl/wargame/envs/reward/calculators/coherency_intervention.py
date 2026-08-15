"""Charge a model for a move the coherency referee had to correct.

**The one gradient that end-of-move enforcement does not supply.** Under
`coherency.enforce_move` every action that gets reverted produces the *identical*
outcome -- the model stays where it was -- so every reverted action has the same
return, the same advantage, and a policy gradient of exactly **zero** inside that
whole equivalence class. Nothing in the reward prefers a legal stay to an illegal
lunge, and the only force acting there is the entropy bonus, which pushes toward
uniform over a class that covers most of the action space.

That is not a theory. Measured with the referee switched off, so the numbers
describe what the policy *chooses* rather than what the board shows:

| | units coherent | models adrift |
|---|---|---|
| trained with `objective_hold.require_coherent`, never enforced | **0.847** | 2.20 |
| the same lineage after 300 epochs *under enforcement* | **0.630** | 5.37 |

Training under the constraint made intent **worse than never training under it**,
and 0.630 is inside the unconstrained control's own range. Enforcement was doing
all the work: the board read 1.000 throughout.

**Why `require_coherent` cannot cover this case.** That gate keys on the model's
position, and the reward is computed *after* enforcement has already made the
board legal -- so under `enforce_move` the gate sees a coherent unit every step
and never fires. It is inert exactly when it is most needed. This term keys on
the intervention instead, which is the only signal that survives the correction.

**It is a penalty, and this project's rule is that magnitude matters more than
sign.** `group_cohesion` at -0.2 inverted the baseline ranking and at -0.05 sits
in a winning config, so the default here is small: an episode with ~6 of 25
models displaced per movement phase and ~20 movement phases pays about
`25 * 0.02` = 0.5 per episode at the default, against a ~10-per-term budget.
Raise it only against a measured intent curve, never against `coherency_rate` --
which under enforcement is 1.000 whatever this term does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.reward.calculators.base import PerModelRewardCalculator

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.reward.step_context import StepContext
    from wargame_rl.wargame.envs.wargame_model import WargameModel


class CoherencyInterventionCalculator(PerModelRewardCalculator):
    """Per-model charge for each move coherency enforcement had to correct."""

    def __init__(self, weight: float = 1.0, penalty: float = -0.02) -> None:
        """
        Args:
            weight: Scales the whole term, as for every calculator.
            penalty: Charged to a model whose move enforcement undid or
                shortened. Must be <= 0 -- a positive value would *pay* for
                illegal moves, which is the failure this term exists to fix.
        """
        super().__init__(weight=weight)
        if penalty > 0.0:
            raise ValueError(f"penalty must be <= 0, got {penalty}")
        self.penalty = penalty

    def calculate(
        self,
        model_idx: int,
        model: WargameModel,
        view: BattleView,
        ctx: StepContext,
    ) -> float:
        """Return this model's charge for being corrected on this step."""
        displaced = ctx.models_displaced_by_enforcement
        if displaced is None or model_idx >= len(displaced):
            return 0.0
        if not model.is_alive or not displaced[model_idx]:
            return 0.0
        return float(self.penalty)
