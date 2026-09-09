"""The fall-back move's after-moving conditions (`09-movement-phase.md`), judged at the unit's close."""

from __future__ import annotations

from collections.abc import Sequence

from wargame_rl.wargame.envs.domain.kernel.entities import WargameModel
from wargame_rl.wargame.envs.domain.movement.unit_moves import (
    touched_enemy_units,
    unit_is_coherent,
)


def fall_back_stands(
    models: Sequence[WargameModel],
    members: Sequence[int],
    alive_enemies: Sequence[WargameModel],
    *,
    engagement_range: float,
    base_diameter: float,
    coherency_nearest: float,
    coherency_furthest: float,
) -> bool:
    """`09-movement-phase.md` § Fall-back: the unit ends unengaged and coherent."""
    if touched_enemy_units(
        models,
        members,
        alive_enemies,
        engagement_range=engagement_range,
        base_diameter=base_diameter,
    ):
        return False
    return unit_is_coherent(models, members, coherency_nearest, coherency_furthest)
