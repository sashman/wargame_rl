"""Reward calculator that rewards the player for net VP gained each step."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.reward.calculators.base import GlobalRewardCalculator

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.reward.step_context import StepContext


class VPGainCalculator(GlobalRewardCalculator):
    """Global net-VP reward, normalized by cap_per_turn (unweighted; the phase
    manager applies the configured weight).

    reward = (player_vp_delta - opponent_vp_delta) / cap_per_turn
    """

    def calculate(self, view: BattleView, ctx: StepContext) -> float:
        # The mission's own number. This used to duck-type its way through
        # three `getattr`s to `params["cap_per_turn"]` and fall back to 15,
        # so a mission that priced objectives differently would have rescaled
        # every reward in the run without a word.
        cap_per_turn = view.config.mission.per_round_cap

        if cap_per_turn <= 0:
            return 0.0

        opponent_vp_delta = float(view.opponent_vp_delta)
        net_vp_delta = float(view.player_vp_delta) - opponent_vp_delta
        return net_vp_delta / float(cap_per_turn)
