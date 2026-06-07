"""Success criteria: player VP greater than opponent VP."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.reward.criteria.base import SuccessCriteria

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.reward.step_context import StepContext


class LastTurnCriteria(SuccessCriteria):
    """Succeeds when player VP is greater than opponent VP.

    This criteria evaluates the current state and succeeds if the player's
    victory points are greater than the opponent's victory points.
    """

    def is_successful(self, view: BattleView, ctx: StepContext) -> bool:
        """Check if player VP is greater than opponent VP."""
        return view.player_vp > view.opponent_vp
