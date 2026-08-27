"""Who gets where first, and which ground was always theirs.

The cheapest module here and the one with most to say. `measure-critic-probe`
found `corr(dV, dVP) ~ 0`: the critic knows redistribution is worth something
and has **no grip on which redistribution pays**. Arrival order is exactly that
missing quantity, and it is a distance and a divide -- no sight, no dice.

It also repairs a number `CLAUDE.md` already flags as broken. The
`measure-objective-split` redistribution ceiling is "deliberately optimistic --
no travel time, no return fire"; charging travel time turns an acknowledged
best case into something a plan can be built on.

⚠ **This overstates reach in three ways, all of them upward.** Coherency binds
the unit -- a 2" chain and a 9" span mean a squad travels together and the
straight line from its centroid is a bound rather than a route. Freezing means
only ~92% of ordered inches are delivered. And nothing here routes around a base
or a table edge: the path is the straight line, so terrain that must be walked
around arrives late. Read an arrival round as "not before this", never as "then".
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

from wargame_rl.wargame.envs.domain.battle_view import BattleView
from wargame_rl.wargame.envs.domain.entities import WargameModel, WargameObjective
from wargame_rl.wargame.envs.types.geometry import Polygon


class Ownership(str, Enum):
    """Which deployment zone an objective sits in.

    The three classes are worth different things and are kept by different
    means: ground in your own zone is cheap to hold and expensive to lose,
    ground in the middle is the pivot, and ground in theirs is a denial target
    rather than a holding one. The rules define no other board regions -- there
    is no "half" and no "centre" -- so these are derived from the zone outlines
    and nothing else. 34 of the 45 real tables have non-rectangular zones and
    `long_edges` splits the SHORT axis, so a board-half rule would mean a
    different thing on every table.
    """

    own_zone = "own"
    contested = "contested"
    hostile = "hostile"


@dataclass(frozen=True, slots=True)
class ObjectiveReach:
    """One objective, and what it costs each side to stand on it."""

    index: int
    location: tuple[float, float]
    ownership: Ownership
    player_rounds: float
    """Rounds for the player's fastest unit to reach it. `inf` if none can."""

    opponent_rounds: float
    player_unit: int | None
    opponent_unit: int | None

    @property
    def contested_margin(self) -> float:
        """Rounds the player arrives *ahead* of the opponent. Negative is behind.

        `inf` and `-inf` are real answers, not errors: they mean one side cannot
        reach the objective at all within any number of rounds, which happens
        when a side has been wiped out.
        """
        return self.opponent_rounds - self.player_rounds


def _unit_centroids(
    models: Sequence[WargameModel],
) -> tuple[np.ndarray, np.ndarray]:
    """`(group_ids, centroids)` over units with at least one living model.

    The centroid rather than the nearest member, because coherency binds the
    unit: the body closest to an objective cannot go there alone, and quoting
    its distance would price a move the rules forbid. This is the same reason
    the scripted opponent advances from its unit centroid.
    """
    groups: dict[int, list[np.ndarray]] = {}
    for model in models:
        if not model.is_alive:
            continue
        groups.setdefault(model.group_id, []).append(
            np.array([float(model.location[0]), float(model.location[1])])
        )
    if not groups:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 2), dtype=float)
    ids = sorted(groups)
    centroids = np.array([np.mean(groups[i], axis=0) for i in ids], dtype=float)
    return np.array(ids, dtype=np.int64), centroids


def _objective_distances(
    centroids: np.ndarray, objectives: Sequence[WargameObjective]
) -> np.ndarray:
    """`(n_units, n_objectives)` distance from each centroid to each objective.

    Measured to the objective's **range surface** -- its outline for an area,
    which is zero inside, and its centre for a marker -- so the answer is
    "distance until I am on it", matching how control is scored. Every objective
    on the real tables is an area.
    """
    if centroids.size == 0 or not objectives:
        return np.zeros((len(centroids), len(objectives)), dtype=float)
    columns = []
    for objective in objectives:
        area = getattr(objective, "area", None)
        if isinstance(area, Polygon):
            columns.append(
                np.array(
                    [area.distance_to_point(float(x), float(y)) for x, y in centroids]
                )
            )
        else:
            centre = np.array(
                [float(objective.location[0]), float(objective.location[1])]
            )
            columns.append(
                np.maximum(
                    np.linalg.norm(centroids - centre, axis=1)
                    - float(objective.radius_size),
                    0.0,
                )
            )
    stacked: np.ndarray = np.column_stack(columns)
    return stacked


def _rounds_to_arrive(distances: np.ndarray, moves: np.ndarray) -> np.ndarray:
    """`ceil(distance / move)` -- whole rounds, because control is scored at one.

    Fractional rounds would say a unit half-arrives, and half a unit on an
    objective controls nothing: control is a headcount evaluated at a scoring
    moment. A unit that cannot move at all never arrives.
    """
    if distances.size == 0:
        return distances
    with np.errstate(divide="ignore", invalid="ignore"):
        rounds: np.ndarray = np.ceil(distances / moves[:, np.newaxis])
    rounds[~np.isfinite(rounds)] = np.inf
    rounds[moves <= 0.0, :] = np.inf
    return rounds


def _classify(
    objective: WargameObjective,
    player_zone: Polygon | None,
    opponent_zone: Polygon | None,
) -> Ownership:
    """Which zone an objective's own position falls in."""
    x, y = float(objective.location[0]), float(objective.location[1])
    if player_zone is not None and player_zone.contains(x, y):
        return Ownership.own_zone
    if opponent_zone is not None and opponent_zone.contains(x, y):
        return Ownership.hostile
    return Ownership.contested


def objective_reach(
    view: BattleView,
    player_moves: np.ndarray,
    opponent_moves: np.ndarray,
) -> tuple[ObjectiveReach, ...]:
    """Earliest arrival on every objective, for both sides, plus its zone.

    `player_moves` and `opponent_moves` are per-**model** normal moves in board
    units (`board.threat.move_reach`); a unit takes its slowest member's, since
    the unit arrives when its last model does.

    ⚠ Objectives are read from the live board, so padded objective slots -- which
    `objective_budget` parks at the board centre -- never appear here. Reading
    them off the padded observation instead would report a phantom objective
    every unit reaches in round one.
    """
    objectives = view.objectives
    player_ids, player_centroids = _unit_centroids(view.player_models)
    opponent_ids, opponent_centroids = _unit_centroids(view.opponent_models)

    player_rounds = _rounds_to_arrive(
        _objective_distances(player_centroids, objectives),
        _unit_moves(view.player_models, player_ids, player_moves),
    )
    opponent_rounds = _rounds_to_arrive(
        _objective_distances(opponent_centroids, objectives),
        _unit_moves(view.opponent_models, opponent_ids, opponent_moves),
    )

    results: list[ObjectiveReach] = []
    for index, objective in enumerate(objectives):
        mine, my_unit = _fastest(player_rounds, player_ids, index)
        theirs, their_unit = _fastest(opponent_rounds, opponent_ids, index)
        results.append(
            ObjectiveReach(
                index=index,
                location=(float(objective.location[0]), float(objective.location[1])),
                ownership=_classify(
                    objective,
                    view.deployment_outline,
                    view.opponent_deployment_outline,
                ),
                player_rounds=mine,
                opponent_rounds=theirs,
                player_unit=my_unit,
                opponent_unit=their_unit,
            )
        )
    return tuple(results)


def _unit_moves(
    models: Sequence[WargameModel], unit_ids: np.ndarray, moves: np.ndarray
) -> np.ndarray:
    """`(n_units,)` the slowest living member's move, per unit.

    The slowest, because the unit arrives when its last model does -- coherency
    will not let the fast half go on ahead.
    """
    if unit_ids.size == 0:
        return np.zeros(0, dtype=float)
    slowest = np.full(len(unit_ids), np.inf, dtype=float)
    position = {int(group): index for index, group in enumerate(unit_ids)}
    for index, model in enumerate(models):
        if not model.is_alive or model.group_id not in position:
            continue
        slot = position[model.group_id]
        slowest[slot] = min(slowest[slot], float(moves[index]))
    slowest[~np.isfinite(slowest)] = 0.0
    return slowest


def _fastest(
    rounds: np.ndarray, unit_ids: np.ndarray, objective: int
) -> tuple[float, int | None]:
    """`(rounds, unit_id)` for the quickest unit to one objective."""
    if rounds.size == 0:
        return float("inf"), None
    column = rounds[:, objective]
    best = int(np.argmin(column))
    if not np.isfinite(column[best]):
        return float("inf"), None
    return float(column[best]), int(unit_ids[best])
