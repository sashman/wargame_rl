from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.reward.criteria.base import SuccessCriteria

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.reward.step_context import StepContext
    from wargame_rl.wargame.envs.wargame import WargameEnv


class AllModelsGroupedCriteria(SuccessCriteria):
    """Succeeds when every model is within ``max_distance`` of at least one
    same-group member.

    Models that are the sole member of their group are considered grouped.
    Requires ``model_model_norms`` in the distance cache.
    """

    def __init__(self, max_distance: float = 10.0) -> None:
        self.max_distance = max_distance

    def is_successful(self, env: WargameEnv, ctx: StepContext) -> bool:
        cache = ctx.distance_cache
        if cache.model_model_norms is None:
            return False

        group_ids = np.array([m.group_id for m in env.wargame_models], dtype=np.intp)
        return cache.all_models_within_group_distance(group_ids, self.max_distance)
