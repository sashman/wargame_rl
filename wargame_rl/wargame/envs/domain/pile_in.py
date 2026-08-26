"""Pile-in step: engaged units close up 3" before blows are traded.

`docs/rules/12-fight-phase.md` § Pile-in step. Both players pile in with any
eligible unit they choose; the active player resolves all of theirs first.

| | |
|---|---|
| **Maximum distance** | 3" |
| **Eligible if** | it is the Fight phase and at least one of: the unit is engaged; it made a charge move this turn |
| **Effect** | the unit moves as described in `03-moving.md` |

**Before moving.** Select pile-in targets: if the unit is engaged, **every**
enemy unit it is engaged with; otherwise one or more enemy units within 5".

**While moving.** Models in base contact with an enemy **cannot be moved**.
Every model that is moved must end **closer to the closest pile-in target**, and
engaged with it if possible.

**After moving.** The unit must be engaged, and every model that started this
move engaged with an enemy unit must still be engaged with that unit.

⚠ **Compulsory and auto-resolved, like consolidate.** The rules let the
controlling player choose *which* eligible units pile in and *which* targets a
disengaged unit selects; the fight phase carries no agent action, so this piles
in every eligible unit toward its nearest target. `DEFERRED: fight.pile_in_choice`.

⚠ **Why it matters and is not cosmetic.** Without it a unit that lands at the
outer edge of engagement range fights from there and can never close, and a unit
whose nearest model dies cannot re-establish contact -- so a lock decays by
attrition rather than by either player's decision. It is also what
`fight.passing` waits on: the rules let a unit whose targets all died wait to
see whether an enemy pile-in brings something within 5", and with no pile-in
there is nothing to wait for.

The move is all-or-nothing at the **unit**, per `03-moving.md`, and deliberately
not routed through `coherency.enforce_move` -- which is off on every shipped
config and would let an illegal pile-in simply stand. Same shape as the charge
referee and the consolidate move.
"""

from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs.domain.coherency import evaluate_coherency
from wargame_rl.wargame.envs.domain.engagement import engagement_matrix
from wargame_rl.wargame.envs.domain.entities import WargameModel
from wargame_rl.wargame.envs.domain.movement import back_off_to_unengaged, resolve_move

# A model whose base already touches an enemy's is "in base contact" and the
# rules pin it. Bases touch at exactly `r_a + r_b`, so a tolerance decides the
# boundary case; without one, float noise pins or frees a model at random.
_CONTACT_TOLERANCE = 1e-6
# How far a disengaged but eligible unit may look for a pile-in target.
SELECTION_RANGE_INCHES = 5.0


def _members(models: list[WargameModel], group: int) -> list[int]:
    """Living members of `group`, in model index order."""
    return [
        index
        for index, model in enumerate(models)
        if model.is_alive and int(model.group_id) == group
    ]


def pile_in(
    models: list[WargameModel],
    enemy_models: list[WargameModel],
    *,
    eligible_units: set[int],
    max_distance: float,
    selection_range: float,
    engagement_range: float,
    base_radius: float,
    board: tuple[float, float],
    coherency_nearest: float,
    coherency_furthest: float,
) -> list[int]:
    """Pile in every eligible unit, and report which group ids moved.

    Args:
        models: the piling-in force, in index order.
        enemy_models: the opposing force; only living models count.
        eligible_units: group ids eligible to fight this phase, captured before
            anybody swings — casualties change engagement, and the rules make a
            unit eligible if it *was* engaged at the start of the step.
        max_distance: the pile-in move, in board units (3" by the rules).
        selection_range: how far a disengaged unit may pick targets (5").
        engagement_range: the engagement predicate's range, in board units.
        base_radius: the moving force's base radius, in board units.
        board: ``(width, height)``, for the edge clamp.
        coherency_nearest: the 2" chain, in board units.
        coherency_furthest: the 9" spread, in board units.

    Returns:
        The group ids whose pile-in stood, in ascending order.
    """
    alive_enemies = [m for m in enemy_models if m.is_alive]
    if not alive_enemies or not eligible_units:
        return []
    lower = np.array([base_radius, base_radius], dtype=float)
    upper = np.array([board[0] - base_radius, board[1] - base_radius], dtype=float)
    enemy_centres = np.array([m.location for m in alive_enemies], dtype=float)
    enemy_radii = np.array([m.base_radius for m in alive_enemies], dtype=float)
    enemy_groups = np.array([int(m.group_id) for m in alive_enemies], dtype=np.intp)

    moved_units: list[int] = []
    for group in sorted(eligible_units):
        members = _members(models, group)
        if not members:
            continue
        before = np.array([models[i].location for i in members], dtype=float)
        contacts = engagement_matrix(
            before,
            enemy_centres,
            np.ones(len(alive_enemies), dtype=bool),
            np.ones(len(members), dtype=bool),
            engagement_range=engagement_range,
            base_diameter=2.0 * base_radius,
        )
        # ⚠ Targets first, and the two cases are NOT the same set. An engaged
        # unit takes EVERY unit it is engaged with -- it may not walk away from
        # one to close on another. A disengaged one picks from within 5".
        engaged_groups = {
            int(enemy_groups[j])
            for j in np.nonzero(np.asarray(contacts).any(axis=0))[0]
        }
        if engaged_groups:
            targets = engaged_groups
        else:
            gaps = (
                np.linalg.norm(
                    before[:, np.newaxis, :] - enemy_centres[np.newaxis, :, :], axis=2
                )
                - enemy_radii[np.newaxis, :]
                - base_radius
            )
            reachable = np.nonzero(gaps.min(axis=0) <= selection_range)[0]
            targets = {int(enemy_groups[j]) for j in reachable}
        if not targets:
            continue
        target_rows = [
            j for j, group_id in enumerate(enemy_groups) if int(group_id) in targets
        ]
        if _move_unit(
            models,
            members,
            before,
            contacts=np.asarray(contacts),
            target_rows=target_rows,
            alive_enemies=alive_enemies,
            enemy_centres=enemy_centres,
            enemy_radii=enemy_radii,
            max_distance=max_distance,
            engagement_range=engagement_range,
            base_radius=base_radius,
            lower=lower,
            upper=upper,
        ) and _is_legal(
            models,
            members,
            before,
            contacts_before=np.asarray(contacts),
            enemy_centres=enemy_centres,
            enemy_groups=enemy_groups,
            engagement_range=engagement_range,
            base_radius=base_radius,
            coherency_nearest=coherency_nearest,
            coherency_furthest=coherency_furthest,
        ):
            moved_units.append(group)
        else:
            for row, index in enumerate(members):
                models[index].location = np.array(before[row], copy=True)
    return moved_units


def _move_unit(
    models: list[WargameModel],
    members: list[int],
    before: np.ndarray,
    *,
    contacts: np.ndarray,
    target_rows: list[int],
    alive_enemies: list[WargameModel],
    enemy_centres: np.ndarray,
    enemy_radii: np.ndarray,
    max_distance: float,
    engagement_range: float,
    base_radius: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> bool:
    """Walk each unpinned member toward its closest pile-in target.

    Sequential in model index order, reading the board live, exactly as
    `ActionHandler.apply` and the consolidate move are: a model must not end on
    ground a squadmate has already taken.
    """
    target_centres = enemy_centres[target_rows]
    target_radii = enemy_radii[target_rows]
    moved = False
    for row, index in enumerate(members):
        model = models[index]
        # ⚠ **Models in base contact cannot be moved.** The rules pin them, and
        # that is what stops a pile-in dragging a locked model off its opponent.
        gaps_to_all = (
            np.linalg.norm(enemy_centres - np.asarray(model.location), axis=1)
            - enemy_radii
            - model.base_radius
        )
        if gaps_to_all.min() <= _CONTACT_TOLERANCE:
            continue
        gaps = (
            np.linalg.norm(target_centres - np.asarray(model.location), axis=1)
            - target_radii
            - model.base_radius
        )
        closest = int(np.argmin(gaps))
        heading = np.asarray(target_centres[closest], dtype=float) - np.asarray(
            model.location, dtype=float
        )
        length = float(np.linalg.norm(heading))
        if length <= 0.0:
            continue
        displacement = heading / length * min(max_distance, length)
        friendly = [
            other
            for position, other in enumerate(models)
            if other.is_alive and position != index
        ]
        friendly_centres = np.array(
            [m.location for m in friendly] or np.empty((0, 2)), dtype=float
        )
        friendly_radii = np.array([m.base_radius for m in friendly], dtype=float)
        target = np.clip(model.location + displacement, lower, upper)
        # ⚠ No `back_off_to_unengaged` ENGAGEMENT rings here, unlike an ordinary
        # move: a pile-in is meant to end in contact. Occupied bases are still
        # passed, so a model may end engaged but never inside another model --
        # the same exemption the charge takes in `ActionHandler.apply`.
        occupied_centres = np.concatenate([enemy_centres, friendly_centres])
        occupied_reach = (
            np.concatenate([enemy_radii, friendly_radii]) + model.base_radius
        )
        resolved = back_off_to_unengaged(
            model.location,
            resolve_move(
                model.location,
                target - model.location,
                model.base_radius,
                enemy_centres,
                enemy_radii,
                friendly_centres,
                friendly_radii,
            ),
            np.empty((0, 2), dtype=float),
            np.empty(0, dtype=float),
            occupied_centres,
            occupied_reach,
        )
        if not np.array_equal(resolved, model.location):
            model.location = resolved
            moved = True
    return moved


def _is_legal(
    models: list[WargameModel],
    members: list[int],
    before: np.ndarray,
    *,
    contacts_before: np.ndarray,
    enemy_centres: np.ndarray,
    enemy_groups: np.ndarray,
    engagement_range: float,
    base_radius: float,
    coherency_nearest: float,
    coherency_furthest: float,
) -> bool:
    """The after-moving conditions, per `12-fight-phase.md` § Pile-in move."""
    after = np.array([models[i].location for i in members], dtype=float)
    contacts_after = np.asarray(
        engagement_matrix(
            after,
            enemy_centres,
            np.ones(len(enemy_centres), dtype=bool),
            np.ones(len(members), dtype=bool),
            engagement_range=engagement_range,
            base_diameter=2.0 * base_radius,
        )
    )
    # "The unit must be engaged."
    if not contacts_after.any():
        return False
    # "Every model that started engaged with an enemy unit must still be
    # engaged with that unit." Per model AND per unit, not merely "still
    # engaged with something" -- a pile-in may not trade one opponent for
    # another.
    for row in range(len(members)):
        started = {int(enemy_groups[j]) for j in np.nonzero(contacts_before[row])[0]}
        if not started:
            continue
        still = {int(enemy_groups[j]) for j in np.nonzero(contacts_after[row])[0]}
        if not started.issubset(still):
            return False
    # A unit move is a unit move: it must still be one body.
    return bool(
        evaluate_coherency(
            positions=after,
            group_ids=np.zeros(len(members), dtype=np.intp),
            alive_mask=np.ones(len(members), dtype=bool),
            base_radii=np.array([models[i].base_radius for i in members], dtype=float),
            nearest_distance=coherency_nearest,
            furthest_distance=coherency_furthest,
        ).all_coherent
    )
