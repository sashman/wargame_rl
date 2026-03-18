"""Reward calculator that rewards the player for VP gained each step."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.reward.calculators.base import GlobalRewardCalculator

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.reward.step_context import StepContext


class VPGainCalculator(GlobalRewardCalculator):
    """Global reward proportional to player VP gained this step.

    Encourages the agent to score victory points. No opponent term;
    reward = weight * player_vp_delta.
    """

    def calculate(self, view: BattleView, ctx: StepContext) -> float:
        return self.weight * float(view.player_vp_delta)
