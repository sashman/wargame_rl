from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.domain.entities import alive_mask_for
from wargame_rl.wargame.envs.reward.calculators.base import PerModelRewardCalculator

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.reward.step_context import StepContext
    from wargame_rl.wargame.envs.wargame_model import WargameModel


class GroupCohesionCalculator(PerModelRewardCalculator):
    """Negative reward proportional to distance beyond ``max_distance``
    from the closest same-group model.

    Returns 0 when the model is within range or is alone in its group.
    """

    def __init__(
        self,
        weight: float = 1.0,
        group_max_distance: float | None = None,
        violation_penalty: float | None = None,
    ) -> None:
        super().__init__(weight=weight)
        self.group_max_distance = group_max_distance
        self.violation_penalty = violation_penalty

    def calculate(
        self,
        model_idx: int,
        model: WargameModel,
        view: BattleView,
        ctx: StepContext,
    ) -> float:
        cache = ctx.distance_cache
        if cache.model_model_norms is None:
            return 0.0

        group_ids = np.array([m.group_id for m in view.player_models], dtype=np.intp)
        alive = alive_mask_for(view.player_models)
        min_dists = cache.min_distances_to_same_group(group_ids, alive_mask=alive)
        min_dist = float(min_dists[model_idx])
        max_distance = float(
            self.group_max_distance if self.group_max_distance is not None else 10.0
        )
        if min_dist <= max_distance:
            return 0.0

        excess = min_dist - max_distance
        penalty = float(
            self.violation_penalty if self.violation_penalty is not None else -10.0
        )
        return penalty * excess

    @property
    def needs_model_model_distances(self) -> bool:
        return True
