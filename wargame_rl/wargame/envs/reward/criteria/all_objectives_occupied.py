from __future__ import annotations

from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.domain.kernel.entities import alive_mask_for
from wargame_rl.wargame.envs.reward.criteria.base import SuccessCriteria

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.reward.step_context import StepContext


class AllObjectivesOccupiedCriteria(SuccessCriteria):
    """Succeeds when EVERY objective has at least ``min_models`` alive player
    models inside it.

    The two criteria on file count models and ignore points:
    ``all_at_objectives`` asks whether every model stands on *some* objective,
    ``fraction_at_objectives`` whether enough of them do. Neither can tell
    twelve models on one point from three on each of four, which is exactly
    the question the curriculum's spread rung (#340 A3) asks -- go to the
    closest point, one squad per point, nobody crosses. This one counts
    points, on the same base-edge test scoring uses (`norms_offset <= radius`,
    the single definition of "on an objective" since 2026-08-22).

    Dead models do not occupy anything; a scenario with no objectives is
    trivially satisfied, as it is for the other two.
    """

    def __init__(self, min_models: int = 1) -> None:
        if min_models < 1:
            raise ValueError(f"min_models must be at least 1, got {min_models}")
        self.min_models = min_models

    def is_successful(self, view: BattleView, ctx: StepContext) -> bool:
        cache = ctx.distance_cache
        if cache.model_obj_norms_offset.shape[1] == 0:
            return True
        alive = alive_mask_for(view.player_models)
        inside = cache.model_obj_norms_offset <= cache.obj_radii
        occupants = (inside & alive[:, None]).sum(axis=0)
        return bool((occupants >= self.min_models).all())
