"""Reward calculator that rewards the player for VP gained each step."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.reward.calculators.base import GlobalRewardCalculator

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.reward.step_context import StepContext


class VPGainCalculator(GlobalRewardCalculator):
    """Global reward normalized by cap_per_turn so max unweighted reward is 1.0.

    reward = weight * (player_vp_delta / cap_per_turn)
    """

    def calculate(self, view: BattleView, ctx: StepContext) -> float:
        cap_per_turn = 15
        config = getattr(view, "config", None)
        mission = getattr(config, "mission", None)
        mission_params = getattr(mission, "params", None)
        if isinstance(mission_params, dict):
            cap_per_turn = int(mission_params.get("cap_per_turn", cap_per_turn))

        if cap_per_turn <= 0:
            return 0.0

        return self.weight * (float(view.player_vp_delta) / float(cap_per_turn))
