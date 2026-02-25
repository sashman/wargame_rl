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

        models = env.wargame_models
        for i, model in enumerate(models):
            same_group_mask = np.array(
                [j != i and m.group_id == model.group_id for j, m in enumerate(models)]
            )
            if not same_group_mask.any():
                continue
            min_dist = float(cache.model_model_norms[i, same_group_mask].min())
            if min_dist > self.max_distance:
                return False

        return True
