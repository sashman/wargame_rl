"""Success criteria: player ahead on VP (player_vp > opponent_vp)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.reward.criteria.base import SuccessCriteria

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.reward.step_context import StepContext


class PlayerAheadOnVPCriteria(SuccessCriteria):
    """Succeeds when the player is ahead on VP (player_vp > opponent_vp)."""

    def is_successful(self, view: BattleView, ctx: StepContext) -> bool:
        return view.player_vp > view.opponent_vp
