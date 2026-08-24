"""Consolidate step: a unit that fought shuffles up to 3" afterwards.

`docs/rules/12-fight-phase.md` § Consolidate step. A unit that **was eligible to
fight this phase** picks a mode, and the modes are **assessed in order and the
first whose conditions are met is compulsory**:

| Mode | Conditions |
|---|---|
| **Ongoing** | The unit is engaged. |
| **Engaging** | Otherwise, within 3" of one or more enemy units. |
| **Objective** | Otherwise, within 3" of one or more objectives. |

⚠ **Only Objective is implemented, and the ORDER is why that is nearly a no-op
rather than a third of the rule.** A unit that charged, survived and is still in
contact is in Ongoing mode, so it never reaches Objective; a unit that wiped its
target but stands within 3" of any other enemy is in Engaging mode, so it never
reaches Objective either. What is left is a unit that fought, is now engaged with
nobody, has no enemy within 3", and is within 3" of an objective — a unit that
killed everything near it. This is scope, taken deliberately: Ongoing and
Engaging both need a *pile-in style* move toward a selected enemy unit, and
Engaging additionally drags fresh enemy units into the fight and grants them a
swing, which needs the alternating activation that v1 does not have.
`DEFERRED: consolidate.ongoing`, `DEFERRED: consolidate.engaging`.

⚠ **Which objective is chosen is the env's, not the agent's.** The rules let the
controlling player select one of the objectives in range; there is no action for
it, so this takes the nearest and records `DEFERRED: consolidate.select_objective`.
A choice worth an action slice is a choice the agent can make differently, and
this one fires on a unit that already has an objective within three inches.

The move itself is all-or-nothing at the **unit**, which is what
`docs/rules/03-moving.md` prescribes when an after-moving condition fails — the
same shape as the charge referee in `env_components/actions.py::_enforce_charge`,
and deliberately not routed through `coherency.enforce_move` (off on every
shipped config), which would let an illegal consolidation simply stand.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from wargame_rl.wargame.envs.domain.coherency import evaluate_coherency
from wargame_rl.wargame.envs.domain.engagement import engagement_matrix
from wargame_rl.wargame.envs.domain.entities import WargameModel, WargameObjective
from wargame_rl.wargame.envs.domain.movement import back_off_to_unengaged, resolve_move

# A model already inside its objective has nothing to gain, and floating point
# makes "exactly on the boundary" a coin toss. Anything under this counts as
# arrived.
_ARRIVED_TOLERANCE = 1e-9
# Walked a hair PAST the objective's edge rather than onto it. The after-moving
# condition is `norms_offset <= radius`, the same strict test scoring uses, and
# stopping exactly on the boundary leaves a float rounding either way to decide
# whether the unit arrived -- which would revert the whole consolidation about
# half the time it succeeded.
_ARRIVAL_MARGIN = 1e-6


def _unit_members(models: list[WargameModel], group: int) -> list[int]:
    return [
        index
        for index, model in enumerate(models)
        if model.is_alive and int(model.group_id) == group
    ]


def _within(
    positions: np.ndarray,
    enemy_positions: np.ndarray,
    *,
    distance: float,
    base_diameter: float,
) -> bool:
    """Is any of these models within `distance` of any living enemy, base to base?

    Reuses the engagement predicate with the 3" consolidation distance in place
    of the engagement range — "within 3 inches of an enemy unit" is the same
    measurement as "engaged", read at a different radius, so it is the same
    function rather than a fourth copy of the base-to-base test.
    """
    if len(enemy_positions) == 0:
        return False
    matrix = engagement_matrix(
        positions,
        enemy_positions,
        np.ones(len(enemy_positions), dtype=bool),
        engagement_range=distance,
        base_diameter=base_diameter,
    )
    return bool(matrix.any())


def _nearest_objective_in_reach(
    offsets: np.ndarray, members: list[int], max_distance: float
) -> int | None:
    """The nearest objective within reach of the unit, or None if there is none.

    Named for what it does. `_select_objective` would have read as the rules'
    *the player selects one of those in range*, which is
    `DEFERRED: consolidate.select_objective` and deliberately absent — and the
    register's own guard in `tests/test_implementation_status.py` fails on that
    name, which is how the collision was found.
    """
    if offsets.shape[1] == 0:
        return None
    per_objective = offsets[members].min(axis=0)
    in_reach = np.nonzero(per_objective <= max_distance)[0]
    if in_reach.size == 0:
        return None
    return int(in_reach[np.argmin(per_objective[in_reach])])


def consolidate_objective(
    models: list[WargameModel],
    enemy_models: list[WargameModel],
    objectives: list[WargameObjective],
    *,
    eligible_units: set[int],
    objective_offsets: Callable[[], np.ndarray],
    max_distance: float,
    engagement_range: float,
    base_radius: float,
    board: tuple[float, float],
    coherency_nearest: float,
    coherency_furthest: float,
) -> list[int]:
    """Move every eligible unit in Objective mode, and report which ones moved.

    Args:
        models: the consolidating force, in index order.
        enemy_models: the opposing force; only living models count.
        objectives: the board's objectives, for their centres and radii.
        eligible_units: group ids that **were eligible to fight this phase**,
            captured before the fight step because casualties change engagement.
        objective_offsets: reads the live board and returns the
            ``(n_models, n_objectives)`` distance from each model's base *edge*
            to each objective — injected rather than computed here so that the
            single definition of "in range of an objective" stays single.
        max_distance: the consolidation move, in board units (3" by the rules).
        board: ``(width, height)``, for the edge clamp.

    Returns the group ids that consolidated. Nothing is computed at all when no
    unit is eligible, so a scenario in which nothing ever fights pays nothing.
    """
    if not eligible_units or not objectives:
        return []
    alive_enemies = [m for m in enemy_models if m.is_alive]
    enemy_positions = np.array(
        [m.location for m in alive_enemies] or np.empty((0, 2)), dtype=float
    )
    base_diameter = 2.0 * base_radius
    radii = np.array([o.radius_size for o in objectives], dtype=float)
    lower = np.array([base_radius, base_radius], dtype=float)
    upper = np.array([board[0] - base_radius, board[1] - base_radius], dtype=float)

    candidates: list[tuple[int, list[int]]] = []
    for group in sorted(eligible_units):
        members = _unit_members(models, group)
        if not members:
            continue
        positions = np.array([models[i].location for i in members], dtype=float)
        # Ongoing, then Engaging. Both are DEFERRED, and both PRE-EMPT Objective
        # rather than falling through to it -- the rules make the first matching
        # mode compulsory, so a unit in Ongoing mode does not get to consolidate
        # onto an objective instead.
        if _within(
            positions,
            enemy_positions,
            distance=engagement_range,
            base_diameter=base_diameter,
        ):
            continue
        if _within(
            positions,
            enemy_positions,
            distance=max_distance,
            base_diameter=base_diameter,
        ):
            continue
        candidates.append((group, members))
    if not candidates:
        return []

    offsets = objective_offsets()
    moved_groups: list[int] = []
    for group, members in candidates:
        objective = _nearest_objective_in_reach(offsets, members, max_distance)
        if objective is None:
            continue
        start = {
            index: np.array(models[index].location, copy=True) for index in members
        }
        if _move_unit(
            models,
            members,
            objectives[objective],
            float(radii[objective]),
            offsets[:, objective],
            alive_enemies=alive_enemies,
            max_distance=max_distance,
            engagement_range=engagement_range,
            base_radius=base_radius,
            lower=lower,
            upper=upper,
        ):
            after = objective_offsets()[:, objective]
            if _is_legal(
                models,
                members,
                start,
                before=offsets[:, objective],
                after=after,
                radius=float(radii[objective]),
                coherency_nearest=coherency_nearest,
                coherency_furthest=coherency_furthest,
            ):
                moved_groups.append(group)
                offsets = objective_offsets()
                continue
            for index in members:
                models[index].location = start[index]
    return moved_groups


def _move_unit(
    models: list[WargameModel],
    members: list[int],
    objective: WargameObjective,
    radius: float,
    before: np.ndarray,
    *,
    alive_enemies: list[WargameModel],
    max_distance: float,
    engagement_range: float,
    base_radius: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> bool:
    """Walk each member toward the objective, at most `max_distance`.

    Sequential in model index order, reading the board live, for the same reason
    `ActionHandler.apply` is: a model must not end on ground a squadmate has
    already taken, and the seeded environment depends on the same board coming
    out of the same inputs.

    A model already in range does not move. The rules only require a *moved*
    model to end in range if it can, and shuffling a model that has arrived buys
    nothing while risking the unit's coherency.
    """
    blocker_centres = np.array(
        [m.location for m in alive_enemies] or np.empty((0, 2)), dtype=float
    )
    blocker_radii = np.array([m.base_radius for m in alive_enemies], dtype=float)
    engagement_reach = (
        blocker_radii + engagement_range + base_radius
        if engagement_range > 0.0 and len(alive_enemies)
        else np.empty(0, dtype=float)
    )
    engagement_centres = (
        blocker_centres if len(engagement_reach) else np.empty((0, 2), dtype=float)
    )
    moved = False
    for index in members:
        model = models[index]
        needed = float(before[index]) - radius
        if needed <= _ARRIVED_TOLERANCE:
            continue
        heading = np.asarray(objective.location, dtype=float) - np.asarray(
            model.location, dtype=float
        )
        length = float(np.linalg.norm(heading))
        if length <= 0.0:
            continue
        displacement = heading / length * min(max_distance, needed + _ARRIVAL_MARGIN)
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
        occupied_centres = np.concatenate([blocker_centres, friendly_centres])
        occupied_reach = (
            np.concatenate([blocker_radii, friendly_radii]) + model.base_radius
        )
        resolved = back_off_to_unengaged(
            model.location,
            resolve_move(
                model.location,
                target - model.location,
                model.base_radius,
                blocker_centres,
                blocker_radii,
                friendly_centres,
                friendly_radii,
            ),
            engagement_centres,
            engagement_reach,
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
    start: dict[int, np.ndarray],
    *,
    before: np.ndarray,
    after: np.ndarray,
    radius: float,
    coherency_nearest: float,
    coherency_furthest: float,
) -> bool:
    """Did this consolidation satisfy every after-moving condition?

    Three, from `12-fight-phase.md` and `03-moving.md`:

    * every model that moved ends **in range of the objective if possible, or
      closer to it if not** — so a model that moved and ended neither in range
      nor nearer fails the unit;
    * **the unit must be within range of the selected objective**, which is the
      point of the mode;
    * the unit is still one body.

    A failure reverts the whole unit rather than the offending model, because a
    partial revert would drop a model back onto ground a squadmate has since
    taken, and because all-or-nothing is what the rules say.
    """
    for index in members:
        if np.array_equal(start[index], models[index].location):
            continue
        arrived = after[index] <= radius
        closer = after[index] < before[index] - _ARRIVED_TOLERANCE
        if not arrived and not closer:
            return False
    if not bool((after[members] <= radius).any()):
        return False
    report = evaluate_coherency(
        positions=np.array([models[i].location for i in members], dtype=float),
        group_ids=np.zeros(len(members), dtype=np.intp),
        alive_mask=np.ones(len(members), dtype=bool),
        base_radii=np.array([models[i].base_radius for i in members], dtype=float),
        nearest_distance=coherency_nearest,
        furthest_distance=coherency_furthest,
    )
    return bool(report.all_coherent)
