"""The charge move's after-moving conditions (`11-charge-phase.md`), judged at the unit's close."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from wargame_rl.wargame.envs.domain.kernel.entities import WargameModel
from wargame_rl.wargame.envs.domain.movement.unit_moves import (
    touched_enemy_units,
    unit_is_coherent,
)

# Float slack on "travelled no further than the roll", as the charge ladder
# quantises to it. Matches the whole-phase facade's tolerance.
CHARGE_REACH_EPSILON = 1e-6


def charge_stands(
    models: Sequence[WargameModel],
    members: Sequence[int],
    alive_enemies: Sequence[WargameModel],
    start_positions: dict[int, np.ndarray],
    *,
    target_group: int,
    reach: float,
    engagement_range: float,
    base_diameter: float,
    coherency_nearest: float,
    coherency_furthest: float,
) -> bool:
    """`11-charge-phase.md`: did this unit's charge, against its DECLARED target, stand.

    Every member declared; nobody travelled further than the roll; the unit
    ends engaged with the declared target and with no other enemy unit; every
    model that moved ended closer to the target; and the unit is still one
    body. A charge failing any clause did not happen.
    """
    if not members or not alive_enemies:
        return False
    for index in members:
        model = models[index]
        if not model.declared_charge:
            return False
        travelled = float(
            np.linalg.norm(
                np.asarray(model.location, dtype=float)
                - np.asarray(start_positions[index], dtype=float)
            )
        )
        if travelled > reach + CHARGE_REACH_EPSILON:
            return False
    touched = touched_enemy_units(
        models,
        members,
        alive_enemies,
        engagement_range=engagement_range,
        base_diameter=base_diameter,
    )
    if touched != {target_group}:
        return False
    target_positions = np.array(
        [m.location for m in alive_enemies if int(m.group_id) == target_group],
        dtype=float,
    )
    for index in members:
        start = np.asarray(start_positions[index], dtype=float)
        end = np.asarray(models[index].location, dtype=float)
        if np.array_equal(start, end):
            continue
        before = float(np.linalg.norm(target_positions - start, axis=1).min())
        after = float(np.linalg.norm(target_positions - end, axis=1).min())
        if after >= before:
            return False
    return unit_is_coherent(models, members, coherency_nearest, coherency_furthest)
