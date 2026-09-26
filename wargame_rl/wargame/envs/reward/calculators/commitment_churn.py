"""A churn cost for the planning stream (#384, amendment 18): the share of
living units the head re-committed this turn before they arrived."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.reward.calculators.base import GlobalRewardCalculator

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.reward.step_context import StepContext


class CommitmentChurnCalculator(GlobalRewardCalculator):
    """Charges the planner for changing a plan its soldiers have not yet met.

    Returns MINUS the share of living units whose ground commitment was
    moved from one objective to another this turn while the unit had not
    arrived at the first (`StepContext.commitment_churn`, kept by the
    commitment state). A first commitment, a re-commit after arrival and
    the env's own retirement of a slot cost nothing, and KEEP is always
    legal on a held slot, so the cost is avoidable at every decision.
    Classed as a state global, so under the head writer it lands on the
    PLANNING stream and never reaches a member. Built because the warm start
    on the half-step (CM8w) peaked at 0.61 / 0.96 / 1.00 and fell to
    0.73 / 0.63 / 0.87 while its planner's re-commits before arrival rose
    from a quarter of squads to two thirds; the soldiers did not change.
    With the layer off (`commitment_churn` None) it pays nothing.
    """

    def calculate(self, view: BattleView, ctx: StepContext) -> float:
        if ctx.commitment_churn is None:
            return 0.0
        return -float(ctx.commitment_churn)
