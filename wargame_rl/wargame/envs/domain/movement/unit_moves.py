"""A unit's move as one thing: did it move, is it whole, what does it touch, put it back."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from wargame_rl.wargame.envs.domain.kernel.entities import WargameModel
from wargame_rl.wargame.envs.domain.movement.coherency import evaluate_coherency
from wargame_rl.wargame.envs.domain.movement.engagement import engagement_matrix


def unit_is_coherent(
    models: Sequence[WargameModel],
    members: Sequence[int],
    nearest_distance: float,
    furthest_distance: float,
) -> bool:
    """Is this unit one body, judged over its living members alone."""
    if not members:
        return True
    report = evaluate_coherency(
        positions=np.array([models[i].location for i in members], dtype=float),
        group_ids=np.zeros(len(members), dtype=np.intp),
        alive_mask=np.ones(len(members), dtype=bool),
        base_radii=np.array([models[i].base_radius for i in members], dtype=float),
        nearest_distance=nearest_distance,
        furthest_distance=furthest_distance,
    )
    return bool(report.all_coherent)


def touched_enemy_units(
    models: Sequence[WargameModel],
    members: Sequence[int],
    alive_enemies: Sequence[WargameModel],
    *,
    engagement_range: float,
    base_diameter: float,
) -> set[int]:
    """Enemy unit ids any of these members is engaged with, on the live board."""
    if not members or not alive_enemies:
        return set()
    contacts = engagement_matrix(
        np.array([models[i].location for i in members], dtype=float),
        np.array([m.location for m in alive_enemies], dtype=float),
        np.ones(len(alive_enemies), dtype=bool),
        np.ones(len(members), dtype=bool),
        engagement_range=engagement_range,
        base_diameter=base_diameter,
    )
    return {
        int(alive_enemies[int(j)].group_id)
        for j in np.nonzero(np.asarray(contacts).any(axis=0))[0]
    }


def unit_moved(
    models: Sequence[WargameModel],
    members: Sequence[int],
    start_positions: dict[int, np.ndarray],
) -> bool:
    """Did any member end somewhere other than where it began."""
    return any(
        not np.array_equal(start_positions[i], models[i].location) for i in members
    )


def revert_unit(
    models: Sequence[WargameModel],
    members: Sequence[int],
    start_positions: dict[int, np.ndarray],
) -> None:
    """The rules' remedy: every model back to where it started. Mutates."""
    for index in members:
        models[index].location = np.array(start_positions[index], copy=True)
