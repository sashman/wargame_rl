"""Unit-close referees: does a unit's move stand, judged once its last model acted.

Under a per-model step a unit's members move one at a time, and the rules
judge the *unit's* move when it ends -- `03-moving.md` returns every model to
where it started when an after-moving condition fails. These are the judgements,
each a pure predicate over the board plus one `revert_unit` that applies the
rules' remedy. They wrap the domain's own checks (`engagement_matrix`,
`evaluate_coherency`, `pile_in.agent_move_is_legal`, `enforce_after_move`) so
the rule a unit is held to here is the rule the whole-phase facade holds it to.

`consolidation_mode` is the one piece of new rules content: `12-fight-phase.md`
§ Consolidate step assesses three modes in order and the first that applies is
compulsory. The whole-phase facade resolves the modes constructively for every
unit at once; the per-model facade has to *state* the mode so a unit can be
told which move it is making.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from wargame_rl.wargame.envs.domain.activation import ConsolidationMode
from wargame_rl.wargame.envs.domain.coherency import evaluate_coherency
from wargame_rl.wargame.envs.domain.coherency_enforcement import (
    CoherencyEnforcement,
    enforce_after_move,
)
from wargame_rl.wargame.envs.domain.engagement import engagement_matrix
from wargame_rl.wargame.envs.domain.entities import WargameModel
from wargame_rl.wargame.envs.domain.pile_in import agent_move_is_legal

# Float slack on "travelled no further than the roll", as the charge ladder
# quantises to it. Matches the whole-phase facade's tolerance.
CHARGE_REACH_EPSILON = 1e-6


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


def short_move_stands(
    models: Sequence[WargameModel],
    members: Sequence[int],
    alive_enemies: Sequence[WargameModel],
    start_positions: dict[int, np.ndarray],
    *,
    selection_range: float,
    engagement_range: float,
    base_radius: float,
    coherency_nearest: float,
    coherency_furthest: float,
) -> bool:
    """A pile-in or consolidation judged by the engine's own `_is_legal`."""
    before = np.array([start_positions[i] for i in members], dtype=float)
    return agent_move_is_legal(
        list(models),
        list(members),
        before,
        list(alive_enemies),
        selection_range=selection_range,
        engagement_range=engagement_range,
        base_radius=base_radius,
        coherency_nearest=coherency_nearest,
        coherency_furthest=coherency_furthest,
    )


def coherency_after_unit_move(
    models: Sequence[WargameModel],
    members: Sequence[int],
    nearest_distance: float,
    furthest_distance: float,
    mode: CoherencyEnforcement,
) -> int:
    """The play-time coherency referee, scoped to the unit that just closed.

    `enforce_after_move` judges a whole force and cascades reverts between
    units; under a per-model step the other units have not moved yet, so the
    judgement is over this unit alone. Returns how many models went back.
    """
    if mode is CoherencyEnforcement.off or not members:
        return 0
    unit = [models[i] for i in members]
    return enforce_after_move(unit, nearest_distance, furthest_distance, mode)


def consolidation_mode(
    models: Sequence[WargameModel],
    members: Sequence[int],
    alive_enemies: Sequence[WargameModel],
    objective_offsets: np.ndarray | None,
    *,
    engagement_range: float,
    base_diameter: float,
    consolidate_distance: float,
) -> ConsolidationMode:
    """`12-fight-phase.md` § Consolidate step: the first applicable mode, compulsory.

    `objective_offsets` is the `(n_models, n_objectives)` distance from each
    model's base edge to each objective, or None when there are no objectives.
    """
    if not members:
        return ConsolidationMode.none
    if touched_enemy_units(
        models,
        members,
        alive_enemies,
        engagement_range=engagement_range,
        base_diameter=base_diameter,
    ):
        return ConsolidationMode.ongoing
    if alive_enemies:
        positions = np.array([models[i].location for i in members], dtype=float)
        enemy_positions = np.array([m.location for m in alive_enemies], dtype=float)
        gaps = (
            np.linalg.norm(
                positions[:, np.newaxis, :] - enemy_positions[np.newaxis, :, :], axis=2
            )
            - base_diameter
        )
        if bool((gaps <= consolidate_distance).any()):
            return ConsolidationMode.engaging
    if objective_offsets is not None and objective_offsets.shape[1] > 0:
        rows = objective_offsets[list(members)]
        if bool((rows <= consolidate_distance).any()):
            return ConsolidationMode.objective
    return ConsolidationMode.none
