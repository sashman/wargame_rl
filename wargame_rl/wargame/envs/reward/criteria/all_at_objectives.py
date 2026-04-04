from __future__ import annotations

from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.domain.entities import alive_mask_for
from wargame_rl.wargame.envs.reward.criteria.base import SuccessCriteria

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.reward.step_context import StepContext


class AllAtObjectivesCriteria(SuccessCriteria):
    """Succeeds when every model is within the radius of at least one objective."""

    def is_successful(self, view: BattleView, ctx: StepContext) -> bool:
        player_alive = alive_mask_for(view.player_models)
        return ctx.distance_cache.all_models_at_objectives(alive_mask=player_alive)
