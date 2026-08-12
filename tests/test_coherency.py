"""Coherency predicate: the two conditions, connectivity, and the edge cases.

The predicate is pure, so these are arranged as plain geometry -- positions in,
verdict out -- with no environment involved.
"""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.envs.domain.coherency import (
    CoherencyReport,
    base_to_base_distances,
    evaluate_coherency,
)

NEAREST = 2.0
FURTHEST = 9.0


def report_for(
    positions: list[tuple[float, float]],
    group_ids: list[int],
    alive: list[bool] | None = None,
    radii: list[float] | None = None,
    nearest: float = NEAREST,
    furthest: float = FURTHEST,
) -> CoherencyReport:
    """Evaluate coherency for a hand-written force."""
    n = len(positions)
    return evaluate_coherency(
        positions=np.array(positions, dtype=float),
        group_ids=np.array(group_ids, dtype=np.intp),
        alive_mask=np.array(alive if alive is not None else [True] * n, dtype=bool),
        base_radii=np.array(radii if radii is not None else [0.0] * n, dtype=float),
        nearest_distance=nearest,
        furthest_distance=furthest,
    )


def test_a_chain_at_the_limit_is_coherent() -> None:
    # Arrange: five models in a line, each exactly 2" from the next. The end
    # models are 8" apart, inside the 9" spread.
    positions = [(float(i) * 2.0, 0.0) for i in range(5)]

    # Act
    report = report_for(positions, [0] * 5)

    # Assert
    assert report.all_coherent
    assert report.in_coherency.all()
    assert report.units[0].max_pairwise_distance == pytest.approx(8.0)


def test_a_chain_that_satisfies_spacing_can_still_breach_spread() -> None:
    # Arrange: six models chained at 2" span 10", past the 9" cap. This is the
    # case the second condition exists for -- every model has a neighbour.
    positions = [(float(i) * 2.0, 0.0) for i in range(6)]

    # Act
    report = report_for(positions, [0] * 6)

    # Assert
    unit = report.units[0]
    assert unit.chain_ok.all()
    assert unit.connected
    assert not unit.spread_ok.all()
    assert not report.all_coherent


def test_two_clusters_each_satisfying_the_chain_rule_are_not_a_unit() -> None:
    # Arrange: two pairs, 5" apart. Every model is within 2" of another model
    # and within 9" of every other -- both stated conditions hold -- yet the
    # unit is in two pieces. This is the counterexample connectivity exists for.
    positions = [(0.0, 0.0), (1.0, 0.0), (5.0, 0.0), (6.0, 0.0)]

    # Act
    report = report_for(positions, [0] * 4)

    # Assert
    unit = report.units[0]
    assert unit.chain_ok.all()
    assert unit.spread_ok.all()
    assert unit.n_components == 2
    assert not unit.coherent


def test_the_detached_models_are_the_ones_out_of_coherency() -> None:
    # Arrange: a body of three and a straggler that is chained to nobody.
    positions = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (7.0, 0.0)]

    # Act
    report = report_for(positions, [0] * 4)

    # Assert: blame lands on the straggler, not on the body it left.
    assert report.in_coherency.tolist() == [True, True, True, False]
    assert report.n_models_out_of_coherency == 1


def test_units_are_judged_independently() -> None:
    # Arrange: unit 0 is tight, unit 1 is scattered beyond both limits.
    positions = [(0.0, 0.0), (1.0, 0.0), (20.0, 20.0), (40.0, 20.0)]

    # Act
    report = report_for(positions, [0, 0, 1, 1])

    # Assert
    assert report.n_units == 2
    assert report.n_units_coherent == 1
    assert report.fraction_units_coherent == pytest.approx(0.5)
    assert report.units[0].coherent
    assert not report.units[1].coherent


def test_a_lone_model_is_always_coherent() -> None:
    # Arrange: the rule binds only "a unit of more than one model".
    # Act
    report = report_for([(0.0, 0.0)], [0])

    # Assert
    assert report.all_coherent
    assert report.units[0].size == 1


def test_dead_models_leave_their_unit() -> None:
    # Arrange: the survivors are tight; the casualty lies far away. A dead model
    # is off the board, so it can neither break spread nor satisfy a chain.
    positions = [(0.0, 0.0), (1.0, 0.0), (50.0, 30.0)]

    # Act
    report = report_for(positions, [0] * 3, alive=[True, True, False])

    # Assert
    assert report.all_coherent
    assert report.units[0].size == 2
    # A dead model is reported as not-in-breach, never as a violation.
    assert report.in_coherency.tolist() == [True, True, True]


def test_a_unit_whose_only_survivor_is_isolated_is_coherent() -> None:
    # Arrange: two of three dead. The last model cannot be out of coherency
    # with itself, which is what stops attrition cascading.
    positions = [(0.0, 0.0), (30.0, 0.0), (60.0, 20.0)]

    # Act
    report = report_for(positions, [0] * 3, alive=[False, False, True])

    # Assert
    assert report.all_coherent
    assert report.n_models_out_of_coherency == 0


def test_distances_are_base_to_base() -> None:
    # Arrange: centres 3" apart with 0.63" bases leaves a 1.74" gap -- inside
    # the 2" chain. The same pair as point models is 3" apart and out of it.
    positions = [(0.0, 0.0), (3.0, 0.0)]

    # Act
    with_bases = report_for(positions, [0, 0], radii=[0.63, 0.63])
    as_points = report_for(positions, [0, 0])

    # Assert
    assert with_bases.all_coherent
    assert not as_points.all_coherent


def test_overlapping_bases_read_as_touching() -> None:
    # Arrange: two models closer together than their radii sum. The gap floors
    # at zero rather than going negative.
    positions = np.array([[0.0, 0.0], [0.5, 0.0]])

    # Act
    gaps = base_to_base_distances(positions, np.array([1.0, 1.0]))

    # Assert
    assert gaps.min() == 0.0
    assert gaps[0, 1] == 0.0


def test_no_live_model_reports_as_coherent() -> None:
    # Arrange: a wiped-out force has no unit to be in breach.
    # Act
    report = report_for([(0.0, 0.0), (1.0, 0.0)], [0, 0], alive=[False, False])

    # Assert
    assert report.all_coherent
    assert report.n_units == 0
    assert report.fraction_units_coherent == 1.0


@pytest.mark.parametrize(
    ("separation", "expected"),
    [(1.99, True), (2.0, True), (2.01, False)],
)
def test_the_chain_boundary_is_inclusive(separation: float, expected: bool) -> None:
    # Arrange: "within 2 inches" includes exactly 2 inches.
    # Act
    report = report_for([(0.0, 0.0), (separation, 0.0)], [0, 0])

    # Assert
    assert report.all_coherent is expected


@pytest.mark.parametrize(
    ("separation", "expected"),
    [(8.99, True), (9.0, True), (9.01, False)],
)
def test_the_spread_boundary_is_inclusive(separation: float, expected: bool) -> None:
    # Arrange: a chain of models 2" apart bridging the two ends, so only the
    # spread condition is ever at stake.
    n_models = int(np.ceil(separation / 2.0)) + 1
    spacing = separation / (n_models - 1)
    positions = [(float(i) * spacing, 0.0) for i in range(n_models)]

    # Act
    report = report_for(positions, [0] * len(positions))

    # Assert
    assert report.units[0].chain_ok.all()
    assert report.all_coherent is expected


def test_the_larger_component_keeps_the_body_when_a_unit_splits() -> None:
    # Arrange: four together, two detached. The pair is out, not the four.
    positions = [(float(i), 0.0) for i in range(4)] + [(20.0, 0.0), (21.0, 0.0)]

    # Act
    report = report_for(positions, [0] * 6, furthest=100.0)

    # Assert
    assert report.units[0].largest_component_size == 4
    assert report.in_coherency.tolist() == [True] * 4 + [False, False]
