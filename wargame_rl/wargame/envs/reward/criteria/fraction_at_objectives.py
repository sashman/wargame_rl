from __future__ import annotations

from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.domain.entities import alive_mask_for
from wargame_rl.wargame.envs.reward.criteria.base import SuccessCriteria

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.battle_view import BattleView
    from wargame_rl.wargame.envs.reward.step_context import StepContext


class FractionAtObjectivesCriteria(SuccessCriteria):
    """Succeeds when at least ``min_fraction`` of alive models are within the
    radius of an objective.

    The all-or-nothing ``all_at_objectives`` scales badly with army size: if each
    model independently has probability ``p`` of being on an objective, success
    needs ``p**n_models``, which is unreachable well before 25 models. This
    criteria degrades gracefully instead, so a curriculum can raise the bar in
    steps.

    Dead models are excluded from numerator and denominator alike, so casualties
    neither help nor hinder — note this differs from ``all_at_objectives``, which
    treats dead models as satisfied.
    """

    def __init__(self, min_fraction: float = 0.5) -> None:
        if not 0.0 < min_fraction <= 1.0:
            raise ValueError(
                f"min_fraction must be in (0, 1], got {min_fraction}. "
                "Use all_at_objectives for a strict all-models requirement."
            )
        self.min_fraction = min_fraction

    def is_successful(self, view: BattleView, ctx: StepContext) -> bool:
        alive = alive_mask_for(view.player_models)
        fraction = ctx.distance_cache.fraction_at_objectives(alive_mask=alive)
        return fraction >= self.min_fraction
