"""Success criteria based on Victory Points."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.reward.criteria.base import SuccessCriteria

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.reward.step_context import StepContext
    from wargame_rl.wargame.envs.wargame import WargameEnv


class PlayerLeadingVPCriteria(SuccessCriteria):
    """Succeeds when the player has more Victory Points than the opponent."""

    def is_successful(self, env: WargameEnv, ctx: StepContext) -> bool:
        player_vp = getattr(env, "player_vp", 0)
        opponent_vp = getattr(env, "opponent_vp", 0)
        return player_vp > opponent_vp
