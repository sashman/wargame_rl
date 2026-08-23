"""Baseline: `squad_march_take`'s preferences, matched optimally instead of greedily.

**This exists to price a ceiling, not to be a good policy.** The open question
it answers is whether the agent's offence deficit is an *allocation* failure at
all. Every shaping proposal aimed at that deficit assumes a better squad ->
objective assignment is available and worth having. If a globally optimal
assignment is worth nothing over the greedy one, the assumption is false and the
proposals are aimed at nothing.

`squad_march_take` is the strongest scripted policy measured here, and its
assignment is **greedy**: sort objectives by ascending opponent count, then walk
that list handing each objective to the nearest squad not yet spoken for. Greedy
matching is the classic place a small local choice compounds -- the nearest squad
to the cheapest objective may be the only squad that can reach the far one in
time, and greedy spends it anyway.

This subclass changes **only** that step. The preference order is identical
(cheap ground first, reinforcement worth progressively less), and so is the
movement, the coherency handling and everything else inherited from
`ScriptedSquadMarchDenyPolicy`. What changes is that the assignment minimises
*total* cost over all squads at once rather than committing one squad at a time.
Any score difference is therefore attributable to greedy-versus-optimal matching
and to nothing else.

The assignment is exact, by dynamic programming over subsets of squads:
``O(n^2 * 2^n)``, which at the eight squads this project runs is ~16k operations
per movement phase. Hungarian would be asymptotically better and is not worth the
dependency at this size. Above `MAX_EXACT_SQUADS` it falls back to the parent's
greedy assignment rather than hanging.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.baseline.registry import register_baseline
from wargame_rl.wargame.envs.baseline.scripted_squad_march_deny import occupants
from wargame_rl.wargame.envs.baseline.scripted_squad_march_take import (
    ScriptedSquadMarchTakePolicy,
)

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.entities import WargameModel, WargameObjective
    from wargame_rl.wargame.envs.wargame import WargameEnv

# Above this many squads the subset DP stops being free; fall back to greedy.
MAX_EXACT_SQUADS = 12
# Taking an objective is worth about a board crossing of travel, so distance
# breaks ties between objectives rather than deciding whether to take one.
OBJECTIVE_VALUE = 60.0
# What each additional squad on an already-claimed objective is worth, relative
# to the one before it. Reinforcing is not worthless -- control is a headcount --
# but it is worth much less than opening a new point.
REINFORCEMENT_DECAY = 0.25
# Charged per enemy model already standing there. Reproduces the parent's
# ascending-opponent-count preference as a cost rather than a sort key.
CONTEST_COST = 12.0


def optimal_assignment(cost: np.ndarray) -> list[int]:
    """Minimum-cost one-to-one assignment of rows to columns, exactly.

    Dynamic programme over subsets of columns: `best[mask]` is the least cost of
    assigning the first `popcount(mask)` rows to exactly the columns in `mask`.
    Rows are assigned in order, so the row index is implied by the population
    count and only the column set has to be carried.

    Args:
        cost: `(n_rows, n_cols)` costs, with `n_cols >= n_rows`.

    Returns:
        For each row, the column assigned to it.
    """
    n_rows, n_cols = cost.shape
    size = 1 << n_cols
    best = np.full(size, np.inf)
    best[0] = 0.0
    choice = np.full(size, -1, dtype=np.int64)

    for mask in range(size):
        row = int(bin(mask).count("1"))
        if row >= n_rows or not np.isfinite(best[mask]):
            continue
        for column in range(n_cols):
            bit = 1 << column
            if mask & bit:
                continue
            candidate = best[mask] + cost[row, column]
            if candidate < best[mask | bit]:
                best[mask | bit] = candidate
                choice[mask | bit] = column

    # Recover the columns by walking the choices back to the empty set.
    final = min(
        (mask for mask in range(size) if bin(mask).count("1") == n_rows),
        key=lambda mask: best[mask],
    )
    columns = [0] * n_rows
    mask = final
    for row in range(n_rows - 1, -1, -1):
        column = int(choice[mask])
        columns[row] = column
        mask &= ~(1 << column)
    return columns


class ScriptedAssignmentOptimalPolicy(ScriptedSquadMarchTakePolicy):
    """`squad_march_take` with a globally optimal squad -> objective matching."""

    def squad_objectives(
        self, models: list[WargameModel], env: WargameEnv, group_ids: list[int]
    ) -> list[WargameObjective]:
        """Match every squad to an objective slot at minimum total cost."""
        objectives = env.objectives
        n_squads = len(group_ids)
        if n_squads == 0 or not objectives:
            return []
        if n_squads > MAX_EXACT_SQUADS:
            return super().squad_objectives(models, env, group_ids)

        opponent_locations = np.array(
            [m.location for m in env.opponent_models if m.is_alive], dtype=float
        )
        if opponent_locations.size == 0:
            opponent_locations = np.empty((0, 2), dtype=float)
        opponent_counts = [occupants(opponent_locations, o) for o in objectives]

        centroids = []
        for group_id in group_ids:
            members = [
                m.location for m in models if m.group_id == group_id and m.is_alive
            ]
            centroids.append(
                np.mean(members, axis=0, dtype=float)
                if members
                else np.zeros(2, dtype=float)
            )

        # One column per (objective, reinforcement rank). Enough ranks that every
        # squad has somewhere to go even when objectives are scarce.
        ranks = -(-n_squads // len(objectives))
        columns: list[tuple[int, int]] = [
            (index, rank) for rank in range(ranks) for index in range(len(objectives))
        ]
        cost = np.zeros((n_squads, len(columns)), dtype=float)
        for squad in range(n_squads):
            for column, (objective_index, rank) in enumerate(columns):
                location = np.asarray(objectives[objective_index].location, dtype=float)
                travel = float(np.linalg.norm(centroids[squad] - location))
                gain = OBJECTIVE_VALUE * (REINFORCEMENT_DECAY**rank) - (
                    CONTEST_COST * opponent_counts[objective_index]
                )
                cost[squad, column] = travel - gain

        assigned = optimal_assignment(cost)
        return [objectives[columns[column][0]] for column in assigned]


register_baseline("assignment_optimal", ScriptedAssignmentOptimalPolicy)
