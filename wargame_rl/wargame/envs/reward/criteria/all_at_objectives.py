from __future__ import annotations

from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.reward.criteria.base import SuccessCriteria

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.reward.step_context import StepContext
    from wargame_rl.wargame.envs.wargame import WargameEnv


class AllAtObjectivesCriteria(SuccessCriteria):
    """Succeeds when every model is within the radius of at least one objective."""

    def is_successful(self, env: WargameEnv, ctx: StepContext) -> bool:
        return ctx.distance_cache.all_models_at_objectives()
