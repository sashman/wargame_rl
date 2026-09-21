"""The commitment state of one seat: what each unit is pointing at (#384).

A commitment is a pointer from a unit to a board token. The GROUND slot holds
one objective index or none; the COMBAT slot a set of up to `K` enemy group
ids in ANY mode (the set retires when one of them dies). Nothing here decides
a commitment -- the env writes the ground slot under
`config.commitments.assignment == "greedy"` (the sticky assignment, B3), a
scripted seat writes what its own assignment says (so the bar has a row on
every readout), and a policy head will write both slots in Stage 1.

Numpy only. The token builder reads `claimants_by_objective`,
`claimants_by_enemy_unit` and `ground_of`; the retimer reads
`committed_objective_per_model`; `scripts/measure_commitments.py` reads the
history the state keeps per turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.env_components.distance_cache import (
    compute_distances,
    objective_counts_from_norms_offset,
)

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.kernel.entities import (
        WargameModel,
        WargameObjective,
    )

NO_TARGET = -1


@dataclass
class UnitCommitment:
    """One unit's two slots. `combat` is ANY-mode: retired when one dies."""

    ground: int = NO_TARGET
    combat: tuple[int, ...] = ()
    # Set at the first turn close on which a member stood inside `ground`;
    # cleared whenever the slot changes. A unit that has arrived keeps its
    # ground commitment through any walk-off (#392): only a LATECOMER to an
    # objective another unit holds is re-assigned.
    arrived: bool = False

    def copy(self) -> UnitCommitment:
        return UnitCommitment(
            ground=self.ground, combat=tuple(self.combat), arrived=self.arrived
        )


def unit_centroids(models: list[WargameModel], groups: list[int]) -> np.ndarray:
    """`(len(groups), 2)` living-member centroids; zeros for a dead unit."""
    out = np.zeros((len(groups), 2), dtype=float)
    for u, g in enumerate(groups):
        members = [
            np.asarray(m.location, dtype=float)
            for m in models
            if int(m.group_id) == g and m.is_alive
        ]
        if members:
            out[u] = np.mean(members, axis=0)
    return out


def greedy_assignment(
    centroids: np.ndarray,
    objective_locations: np.ndarray,
    opponent_counts: np.ndarray,
) -> list[int]:
    """The scripted bar's rule (`ScriptedSquadMarchTakePolicy.squad_objectives`),
    as a pure function: objectives in ascending (opponent count, index) order,
    each taking the nearest unassigned unit; leftover units take the
    objectives in that same order, wrapping. One objective index per unit.
    """
    n_units = int(centroids.shape[0])
    n_obj = int(objective_locations.shape[0])
    if n_units == 0 or n_obj == 0:
        return [NO_TARGET] * n_units
    order = sorted(range(n_obj), key=lambda i: (int(opponent_counts[i]), i))
    unassigned = list(range(n_units))
    targets = [NO_TARGET] * n_units
    for objective_index in order:
        if not unassigned:
            break
        location = objective_locations[objective_index]
        nearest = min(
            unassigned,
            key=lambda u: float(np.linalg.norm(centroids[u] - location)),
        )
        targets[nearest] = objective_index
        unassigned.remove(nearest)
    for u in unassigned:
        targets[u] = order[u % len(order)]
    return targets


@dataclass
class CommitmentState:
    """Per seat: a `UnitCommitment` per group id, and a per-turn history."""

    groups: list[int]
    combat_set_size: int = 2
    by_group: dict[int, UnitCommitment] = field(default_factory=dict)
    # One entry per (battle_round, seat turn) the env closes: a copy of
    # `by_group`, for the persistence readout. The env appends at each close.
    history: list[dict[int, UnitCommitment]] = field(default_factory=list)

    def __post_init__(self) -> None:
        for g in self.groups:
            self.by_group.setdefault(int(g), UnitCommitment())

    # ------------------------------------------------------------ lifecycle
    def clear(self) -> None:
        for g in self.groups:
            self.by_group[int(g)] = UnitCommitment()
        self.history = []

    def record_turn(self) -> None:
        self.history.append({g: c.copy() for g, c in self.by_group.items()})

    # ------------------------------------------------------------- writers
    def set_ground(self, group: int, objective: int) -> None:
        slot = self.by_group[int(group)]
        if slot.ground != int(objective):
            slot.arrived = False
        slot.ground = int(objective)

    def clear_ground(self, group: int) -> None:
        slot = self.by_group[int(group)]
        slot.ground = NO_TARGET
        slot.arrived = False

    def set_combat(self, group: int, targets: tuple[int, ...]) -> None:
        if len(targets) > self.combat_set_size:
            raise ValueError(
                f"a combat set holds at most {self.combat_set_size} targets, "
                f"got {len(targets)}"
            )
        self.by_group[int(group)].combat = tuple(int(t) for t in targets)

    # ------------------------------------------------------------- readers
    def ground_of(self, group: int) -> int:
        return self.by_group[int(group)].ground

    def combat_of(self, group: int) -> tuple[int, ...]:
        return self.by_group[int(group)].combat

    def any_set(self) -> bool:
        return any(c.ground != NO_TARGET or c.combat for c in self.by_group.values())

    def claimants_by_objective(self, n_objectives: int) -> np.ndarray:
        """Expected claimants per objective: a ground commitment counts 1."""
        out = np.zeros(n_objectives, dtype=float)
        for c in self.by_group.values():
            if 0 <= c.ground < n_objectives:
                out[c.ground] += 1.0
        return out

    def claimants_by_enemy_unit(self, enemy_groups: np.ndarray) -> np.ndarray:
        """Expected claimants per enemy unit: an ANY member counts 1/|set|."""
        out = np.zeros(len(enemy_groups), dtype=float)
        index = {int(g): i for i, g in enumerate(enemy_groups)}
        for c in self.by_group.values():
            if not c.combat:
                continue
            share = 1.0 / len(c.combat)
            for t in c.combat:
                if t in index:
                    out[index[t]] += share
        return out

    def committed_objective_per_model(self, models: list[WargameModel]) -> np.ndarray:
        """`(n_models,)` int: each model's unit's ground objective, or -1."""
        return np.array(
            [
                self.by_group.get(int(m.group_id), UnitCommitment()).ground
                for m in models
            ],
            dtype=np.intp,
        )


# ------------------------------------------------------------ the env's writer


def objective_counts_both_sides(
    own: list[WargameModel],
    enemies: list[WargameModel],
    objectives: list[WargameObjective],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(own counts, enemy counts, own model-objective norms) under the scoring
    rule -- the single definition of "on an objective"."""
    own_alive = np.array([m.is_alive for m in own], dtype=bool)
    own_cache = compute_distances(own, objectives, alive_mask=own_alive)
    own_counts = objective_counts_from_norms_offset(
        own_cache.model_obj_norms_offset, own_cache.obj_radii
    )
    if enemies:
        enemy_alive = np.array([m.is_alive for m in enemies], dtype=bool)
        enemy_cache = compute_distances(enemies, objectives, alive_mask=enemy_alive)
        enemy_counts = objective_counts_from_norms_offset(
            enemy_cache.model_obj_norms_offset, enemy_cache.obj_radii
        )
    else:
        enemy_counts = np.zeros(len(objectives), dtype=np.intp)
    return own_counts, enemy_counts, own_cache.model_obj_norms_offset


def assign_greedy(
    state: CommitmentState,
    own: list[WargameModel],
    enemies: list[WargameModel],
    objectives: list[WargameObjective],
) -> None:
    """Write the sticky assignment for every living unit (deployment)."""
    groups = [
        g for g in state.groups if any(m.is_alive and int(m.group_id) == g for m in own)
    ]
    if not groups or not objectives:
        return
    _own_counts, enemy_counts, _norms = objective_counts_both_sides(
        own, enemies, objectives
    )
    centroids = unit_centroids(own, groups)
    locations = np.array([o.location for o in objectives], dtype=float)
    for g, target in zip(groups, greedy_assignment(centroids, locations, enemy_counts)):
        state.set_ground(g, target)


def assign_rotated(
    state: CommitmentState,
    own: list[WargameModel],
    enemies: list[WargameModel],
    objectives: list[WargameObjective],
) -> None:
    """The greedy assignment shifted one unit along in group order (#393).

    Every unit is assigned the objective the greedy rule gave the NEXT unit,
    so on a shape where greedy gives every unit a distinct objective (the A3
    column) no unit is assigned its own greedy pick, and a policy that walks
    to its nearest objective cannot satisfy `all_units_on_commitment`. This
    is the legibility rung's writer: the assignment is only knowable from
    the marked-target relation. With one unit or one objective it is the
    greedy assignment.
    """
    groups = [
        g for g in state.groups if any(m.is_alive and int(m.group_id) == g for m in own)
    ]
    if not groups or not objectives:
        return
    _own_counts, enemy_counts, _norms = objective_counts_both_sides(
        own, enemies, objectives
    )
    centroids = unit_centroids(own, groups)
    locations = np.array([o.location for o in objectives], dtype=float)
    targets = greedy_assignment(centroids, locations, enemy_counts)
    if len(targets) > 1:
        targets = targets[1:] + targets[:1]
    for g, target in zip(groups, targets):
        state.set_ground(g, target)


def retire_and_reassign(
    state: CommitmentState,
    own: list[WargameModel],
    enemies: list[WargameModel],
    objectives: list[WargameObjective],
) -> list[int]:
    """The ground slot's retirement under the greedy writer, at a turn close.

    A unit's ground commitment retires when the unit is dead, or when its
    objective is held by us WITHOUT any of its own living members inside it
    AND the unit has never arrived there (another unit holds it and this one
    is a latecomer). A unit that has ever had a member inside its objective
    keeps the commitment through any walk-off (#392): Stage 0 re-assigned
    the walker for free and the leaving step paid nothing on the six-squad,
    five-objective shape, where two squads share an objective from
    deployment. A retired living unit is
    re-assigned to the nearest objective that is neither ours nor claimed by
    another unit; failing that the nearest not ours; failing that it keeps
    what it had (every objective is ours). Returns the groups re-assigned.
    """
    if not objectives:
        return []
    own_counts, enemy_counts, norms = objective_counts_both_sides(
        own, enemies, objectives
    )
    ours = own_counts > enemy_counts
    radii = compute_distances(own, objectives).obj_radii
    locations = np.array([o.location for o in objectives], dtype=float)
    reassigned: list[int] = []
    for g in state.groups:
        members = [i for i, m in enumerate(own) if int(m.group_id) == g and m.is_alive]
        if not members:
            state.clear_ground(g)
            continue
        current = state.ground_of(g)
        if current == NO_TARGET:
            continue
        inside = any(norms[i, current] <= radii[current] for i in members)
        slot = state.by_group[g]
        if inside:
            slot.arrived = True
        if not (bool(ours[current]) and not inside and not slot.arrived):
            continue
        claimed = {
            c.ground
            for h, c in state.by_group.items()
            if h != g and c.ground != NO_TARGET
        }
        centroid = unit_centroids(own, [g])[0]
        distances = np.linalg.norm(locations - centroid, axis=1)
        free = [k for k in range(len(objectives)) if not ours[k] and k not in claimed]
        if not free:
            free = [k for k in range(len(objectives)) if not ours[k]]
        if not free:
            continue
        state.set_ground(g, int(min(free, key=lambda k: float(distances[k]))))
        reassigned.append(g)
    return reassigned


__all__ = [
    "NO_TARGET",
    "CommitmentState",
    "UnitCommitment",
    "assign_greedy",
    "assign_rotated",
    "greedy_assignment",
    "objective_counts_both_sides",
    "retire_and_reassign",
    "unit_centroids",
]
