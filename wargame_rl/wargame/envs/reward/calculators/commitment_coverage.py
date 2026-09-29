"""A plan-shape term for the planning stream (#384, amendment 13): the
fraction of objectives that at least one living unit is COMMITTED to."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.reward.calculators.base import GlobalRewardCalculator

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.reward.step_context import StepContext


class CommitmentCoverageCalculator(GlobalRewardCalculator):
    """Pays the planner for the shape of its plan, not for its execution.

    Returns the share of the board's objectives that some living unit's
    ground commitment names, read from `StepContext.committed_objective`
    (one entry per model, -1 for none). A plan that commits every squad to
    one objective reads 1/n; a covering plan reads 1.0. Classed as a state
    global, so under the head writer it lands on the PLANNING stream with
    the outcome terms and never reaches a member (the two-stream reward's
    second constraint: the members' pay is the plan's execution alone).
    Built because the outcome stream, broadcast to every unit, paid a
    stacking head the same at every unit and two heads of three never left
    a stack (CM5b) -- and because forking those games showed the outcome
    cannot see a spread squad under members who take five turns to arrive.
    With the layer off (`committed_objective` None) it pays nothing.
    """

    def calculate(self, view: BattleView, ctx: StepContext) -> float:
        committed = ctx.committed_objective
        n_obj = len(view.objectives)
        if committed is None or n_obj == 0:
            return 0.0
        alive = [i for i, m in enumerate(view.player_models) if m.is_alive]
        claimed = {int(committed[i]) for i in alive if 0 <= int(committed[i]) < n_obj}
        return float(len(claimed)) / float(n_obj)
