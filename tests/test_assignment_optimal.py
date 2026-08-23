"""The optimal-assignment baseline: exactness, and that it is a real alternative.

`assignment_optimal` exists to price a ceiling -- how much a *globally optimal*
squad -> objective matching is worth over `squad_march_take`'s greedy one. That
number is only meaningful if the matching really is optimal and really does
differ from greedy, so both are pinned here.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from wargame_rl.wargame.envs.baseline.registry import get_registry
from wargame_rl.wargame.envs.baseline.scripted_assignment_optimal import (
    ScriptedAssignmentOptimalPolicy,
    optimal_assignment,
)


@pytest.mark.parametrize("trial", range(25))
def test_assignment_matches_brute_force(trial: int) -> None:
    """The subset DP finds the same minimum an exhaustive search does."""
    generator = np.random.default_rng(trial)
    n_rows = int(generator.integers(1, 5))
    n_cols = int(generator.integers(n_rows, 7))
    cost = generator.random((n_rows, n_cols)) * 10.0

    columns = optimal_assignment(cost)
    achieved = sum(cost[row, columns[row]] for row in range(n_rows))
    best = min(
        sum(cost[row, permutation[row]] for row in range(n_rows))
        for permutation in itertools.permutations(range(n_cols), n_rows)
    )

    assert achieved == pytest.approx(best)
    assert len(set(columns)) == n_rows, "a column cannot take two squads"


def test_assignment_beats_greedy_where_greedy_is_wrong() -> None:
    """The classic greedy trap: the cheapest first pick strands the second row.

    Greedy assigns row 0 to column 0 (cost 1) and is then forced to pay 100 for
    row 1. Taking the globally optimal pair costs 4. If this ever came back
    equal to greedy the ceiling measurement would be vacuous.
    """
    cost = np.array([[1.0, 2.0], [2.0, 100.0]])

    columns = optimal_assignment(cost)

    assert columns == [1, 0]
    assert sum(cost[row, columns[row]] for row in range(2)) == pytest.approx(4.0)


def test_policy_is_registered_and_subclasses_take() -> None:
    """It is reachable by name and inherits `squad_march_take`'s movement."""
    from wargame_rl.wargame.envs.baseline.scripted_squad_march_take import (
        ScriptedSquadMarchTakePolicy,
    )

    registry = get_registry()

    assert registry["assignment_optimal"] is ScriptedAssignmentOptimalPolicy
    assert issubclass(ScriptedAssignmentOptimalPolicy, ScriptedSquadMarchTakePolicy)
