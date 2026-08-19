"""End-of-move coherency enforcement: the two reverts, and the no-op default.

`03-moving.md` § Making a move undoes a move that ends a unit out of coherency.
These pin both modes against hand-placed geometry, so the revert is checked on
positions rather than on whether a policy happened to break formation.
"""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.envs.domain.coherency import evaluate_coherency
from wargame_rl.wargame.envs.domain.coherency_enforcement import (
    CoherencyEnforcement,
    enforce_after_move,
)
from wargame_rl.wargame.envs.domain.entities import WargameModel
from wargame_rl.wargame.envs.domain.value_objects import position

NEAREST = 2.0
FURTHEST = 9.0


def make_unit(
    starts: list[tuple[float, float]],
    ends: list[tuple[float, float]],
    group_ids: list[int] | None = None,
) -> list[WargameModel]:
    """Build models that have already moved from *starts* to *ends*."""
    group_ids = group_ids if group_ids is not None else [0] * len(starts)
    models = []
    for start, end, group_id in zip(starts, ends, group_ids):
        model = WargameModel(
            location=position(*end),
            distances_to_objectives=np.zeros((1, 2)),
            stats={"current_wounds": 1, "max_wounds": 1},
            group_id=group_id,
        )
        model.previous_location = np.array(start, dtype=model.location.dtype)
        models.append(model)
    return models


def locations(models: list[WargameModel]) -> list[tuple[float, float]]:
    """Current positions, as plain tuples."""
    return [(float(m.location[0]), float(m.location[1])) for m in models]


def is_coherent(models: list[WargameModel]) -> bool:
    """Whether every unit satisfies the whole rule where the models now stand.

    The revert's actual contract. Counting how many models went back says
    nothing about whether the position it left is legal -- which is exactly how
    a `revert_model` that enforced nothing passed its tests.
    """
    report = evaluate_coherency(
        positions=np.array([m.location for m in models], dtype=float),
        group_ids=np.array([m.group_id for m in models], dtype=np.intp),
        alive_mask=np.array([m.is_alive for m in models], dtype=bool),
        base_radii=np.array([m.base_radius for m in models], dtype=float),
        nearest_distance=NEAREST,
        furthest_distance=FURTHEST,
    )
    return all(unit.coherent for unit in report.units)


def test_a_legal_move_is_left_alone() -> None:
    # Arrange: the unit advances 5 and stays chained the whole way.
    starts = [(0.0, 0.0), (1.5, 0.0), (3.0, 0.0)]
    ends = [(5.0, 0.0), (6.5, 0.0), (8.0, 0.0)]
    models = make_unit(starts, ends)

    # Act
    reverted = enforce_after_move(
        models, NEAREST, FURTHEST, CoherencyEnforcement.revert_unit
    )

    # Assert
    assert reverted == 0
    assert locations(models) == ends


def test_revert_unit_sends_the_whole_unit_back() -> None:
    # Arrange: two models advance, the third is left 20 behind and breaks the
    # chain. The spec cancels the *unit's* move, not just the straggler's.
    starts = [(0.0, 0.0), (1.5, 0.0), (3.0, 0.0)]
    ends = [(20.0, 0.0), (21.5, 0.0), (3.0, 0.0)]
    models = make_unit(starts, ends)

    # Act
    reverted = enforce_after_move(
        models, NEAREST, FURTHEST, CoherencyEnforcement.revert_unit
    )

    # Assert: all three, including the two that moved legally.
    assert reverted == 3
    assert locations(models) == starts


def test_revert_model_sends_back_only_the_detached() -> None:
    # Arrange: a straggler that breaks the *chain* while staying inside the
    # spread cap -- 6.5 from the nearest, 8 from the furthest, both under 9.
    # That isolation matters: a straggler far enough to breach spread as well
    # puts every model in the unit in breach, and then the two modes coincide.
    # The pair advances only 2, so sending the straggler home leaves the unit
    # coherent -- which is what lets the local revert stand.
    starts = [(0.0, 0.0), (1.5, 0.0), (3.0, 0.0)]
    ends = [(2.0, 0.0), (3.5, 0.0), (10.0, 0.0)]
    models = make_unit(starts, ends)

    # Act
    reverted = enforce_after_move(
        models, NEAREST, FURTHEST, CoherencyEnforcement.revert_model
    )

    # Assert: the straggler goes back, the pair keeps the ground it took.
    assert reverted == 1
    assert locations(models)[:2] == [(2.0, 0.0), (3.5, 0.0)]
    assert models[2].location.tolist() == [3.0, 0.0]
    assert is_coherent(models), "a local revert that leaves a break is no revert"


def test_revert_model_escalates_when_the_local_revert_does_not_work() -> None:
    # Arrange: the same shape as the test above, but the pair has run 20 away.
    # Sending the straggler home now leaves the unit spanning 3 to 21.5 -- a
    # spread of 18.5 against a cap of 9 -- so the local revert has enforced
    # nothing and must widen to the whole unit.
    #
    # Regression: one pass of `revert_model` gave the same rate of units broken
    # by their own move as running with no enforcement at all (0.024 v 0.025),
    # because selection never re-checked what the revert had done.
    starts = [(0.0, 0.0), (1.5, 0.0), (3.0, 0.0)]
    ends = [(20.0, 0.0), (21.5, 0.0), (15.0, 0.0)]
    models = make_unit(starts, ends)

    # Act
    reverted = enforce_after_move(
        models, NEAREST, FURTHEST, CoherencyEnforcement.revert_model
    )

    # Assert
    assert reverted == 3
    assert locations(models) == starts
    assert is_coherent(models)


def test_revert_model_leaves_a_unit_broken_before_it_moved() -> None:
    # Arrange: a unit already split when its move began -- a casualty took the
    # middle of the chain -- which no revert can repair, because reverting is
    # not a move. The guarantee is "this move did not break the unit", not
    # "the unit is coherent"; closing this break is attrition's job.
    starts = [(0.0, 0.0), (30.0, 0.0)]
    ends = [(1.0, 0.0), (31.0, 0.0)]
    models = make_unit(starts, ends)

    # Act
    reverted = enforce_after_move(
        models, NEAREST, FURTHEST, CoherencyEnforcement.revert_model
    )

    # Assert: it goes back to the start and stops there rather than looping.
    assert reverted == 2
    assert locations(models) == starts
    assert not is_coherent(models)


def test_revert_unit_cancels_the_same_move_entirely() -> None:
    # Arrange: the control for the test above, same geometry, other mode. This
    # pair is the whole difference between the modes.
    starts = [(0.0, 0.0), (1.5, 0.0), (3.0, 0.0)]
    ends = [(20.0, 0.0), (21.5, 0.0), (15.0, 0.0)]
    models = make_unit(starts, ends)

    # Act
    reverted = enforce_after_move(
        models, NEAREST, FURTHEST, CoherencyEnforcement.revert_unit
    )

    # Assert
    assert reverted == 3
    assert locations(models) == starts


def test_a_straggler_that_also_breaches_spread_condemns_the_whole_unit() -> None:
    # Arrange: the same shape, but the pair runs 17 clear. Now no model is
    # within 9 of every other, so `revert_model` reverts everyone too -- it
    # sends back the models *in breach*, and a spread breach is collective.
    # Worth pinning: it is the reason the two modes tie on some geometries and
    # differ by 50 vp on `split_evenly`.
    starts = [(0.0, 0.0), (1.5, 0.0), (3.0, 0.0)]
    ends = [(20.0, 0.0), (21.5, 0.0), (3.0, 0.0)]
    models = make_unit(starts, ends)

    # Act
    reverted = enforce_after_move(
        models, NEAREST, FURTHEST, CoherencyEnforcement.revert_model
    )

    # Assert
    assert reverted == 3
    assert locations(models) == starts


def test_off_touches_nothing() -> None:
    # Arrange: a move that is plainly illegal.
    starts = [(0.0, 0.0), (1.5, 0.0)]
    ends = [(0.0, 0.0), (40.0, 0.0)]
    models = make_unit(starts, ends)

    # Act
    reverted = enforce_after_move(models, NEAREST, FURTHEST, CoherencyEnforcement.off)

    # Assert
    assert reverted == 0
    assert locations(models) == ends


def test_units_are_enforced_independently() -> None:
    # Arrange: unit 0 breaks, unit 1 does not.
    starts = [(0.0, 0.0), (1.5, 0.0), (30.0, 0.0), (31.5, 0.0)]
    ends = [(0.0, 0.0), (25.0, 0.0), (32.0, 0.0), (33.5, 0.0)]
    models = make_unit(starts, ends, group_ids=[0, 0, 1, 1])

    # Act
    enforce_after_move(models, NEAREST, FURTHEST, CoherencyEnforcement.revert_unit)

    # Assert: unit 1 keeps its legal advance.
    assert locations(models)[:2] == starts[:2]
    assert locations(models)[2:] == ends[2:]


def test_a_model_that_did_not_move_has_nothing_to_revert() -> None:
    # Arrange: a unit broken by something other than its own move -- the case
    # the spec reserves End-of-Turn attrition for. `previous_location` is None
    # on a model the action handler never displaced.
    models = make_unit([(0.0, 0.0), (40.0, 0.0)], [(0.0, 0.0), (40.0, 0.0)])
    for model in models:
        model.previous_location = None

    # Act
    reverted = enforce_after_move(
        models, NEAREST, FURTHEST, CoherencyEnforcement.revert_unit
    )

    # Assert: the breach is real and the revert cannot fix it. Measured at ~9%
    # of unit-steps with the move rule on, which is what attrition is for.
    assert reverted == 0


def test_a_dead_model_cannot_break_its_unit() -> None:
    # Arrange: a casualty lying far away while the survivors advance together.
    starts = [(0.0, 0.0), (1.5, 0.0), (50.0, 30.0)]
    ends = [(5.0, 0.0), (6.5, 0.0), (50.0, 30.0)]
    models = make_unit(starts, ends)
    models[2].take_damage(1)

    # Act
    reverted = enforce_after_move(
        models, NEAREST, FURTHEST, CoherencyEnforcement.revert_unit
    )

    # Assert: the living pair keeps its move.
    assert reverted == 0
    assert locations(models)[:2] == ends[:2]


@pytest.mark.parametrize(
    "mode", [CoherencyEnforcement.revert_unit, CoherencyEnforcement.revert_model]
)
def test_a_split_unit_is_caught_by_both_modes(mode: CoherencyEnforcement) -> None:
    # Arrange: two pairs 5 apart. Both stated conditions hold -- every model has
    # a partner within 2, nothing exceeds 9 -- and the unit is still in two
    # pieces. Enforcement has to key on connectivity or this move stands.
    starts = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
    ends = [(0.0, 0.0), (1.0, 0.0), (6.0, 0.0), (7.0, 0.0)]
    models = make_unit(starts, ends)

    # Act
    reverted = enforce_after_move(models, NEAREST, FURTHEST, mode)

    # Assert
    assert reverted > 0


def overlapping_pairs(models: list[WargameModel]) -> list[tuple[int, int]]:
    """Indices of every pair of live models whose bases intersect."""
    pairs = []
    for i, a in enumerate(models):
        for j, b in enumerate(models[i + 1 :], start=i + 1):
            if not (a.is_alive and b.is_alive):
                continue
            gap = float(np.hypot(*(a.location - b.location)))
            if gap < a.base_radius + b.base_radius:
                pairs.append((i, j))
    return pairs


def test_a_revert_never_leaves_two_bases_overlapping() -> None:
    # Arrange: the case that makes a naive revert illegal. Model 1 legally
    # advanced onto the ground model 0 vacated -- resolution runs in index
    # order against live positions, so that is a move the handler permits.
    # Model 0's unit then breaks coherency and it is sent back, onto model 1.
    starts = [(10.0, 0.0), (8.0, 0.0), (11.5, 0.0), (10.0, 30.0)]
    ends = [(40.0, 0.0), (10.0, 0.0), (11.5, 0.0), (10.0, 30.0)]
    models = make_unit(starts, ends, group_ids=[0, 1, 1, 0])
    for model in models:
        model.base_radius = 0.63
    assert not overlapping_pairs(models), "the moved state must start legal"

    # Act
    enforce_after_move(models, NEAREST, FURTHEST, CoherencyEnforcement.revert_model)

    # Assert: model 0 went back, and model 1 was displaced by it and went back
    # too, rather than being left underneath. "No model is left on top of
    # another" is checked in the same breath as coherency (03-moving.md).
    assert models[0].location.tolist() == [10.0, 0.0]
    assert not overlapping_pairs(models)


def test_the_cascade_terminates_at_the_starting_configuration() -> None:
    # Arrange: a chain where every model advanced onto its neighbour's vacated
    # ground, so reverting any one of them must unwind the whole column. The
    # worst case has to land on the pre-move configuration, which is legal.
    starts = [(float(i) * 1.5, 0.0) for i in range(5)]
    ends = [(float(i) * 1.5 + 1.5, 0.0) for i in range(4)] + [(60.0, 0.0)]
    models = make_unit(starts, ends, group_ids=[0, 0, 0, 0, 1])
    for model in models:
        model.base_radius = 0.63

    # Act
    enforce_after_move(models, NEAREST, FURTHEST, CoherencyEnforcement.revert_model)

    # Assert
    assert not overlapping_pairs(models)


def coherent(models: list[WargameModel]) -> bool:
    """True while every unit in the force satisfies the whole rule."""
    return evaluate_coherency(
        positions=np.array([m.location for m in models], dtype=float),
        group_ids=np.array([m.group_id for m in models], dtype=np.intp),
        alive_mask=np.array([m.is_alive for m in models], dtype=bool),
        base_radii=np.array([m.base_radius for m in models], dtype=float),
        nearest_distance=NEAREST,
        furthest_distance=FURTHEST,
    ).all_coherent


def test_repair_gathers_the_stray_and_keeps_everyone_elses_move() -> None:
    """`repair` makes the nearest legal move where `revert_unit` makes none.

    This is the whole point of the mode. The spec's revert cancels five models'
    moves because one of them strayed, which destroys 48.9% of all intended
    movement and cancels 33.1% of unit-moves. Repair pulls the stray back to the
    body and lets the other four keep the ground they took.
    """
    # Four models advance in formation; the fifth overshoots and breaks the chain.
    starts = [(10.0, 10.0), (11.5, 10.0), (13.0, 10.0), (14.5, 10.0), (16.0, 10.0)]
    ends = [(12.0, 10.0), (13.5, 10.0), (15.0, 10.0), (16.5, 10.0), (26.0, 10.0)]

    reverted = make_unit(starts, ends)
    enforce_after_move(reverted, NEAREST, FURTHEST, CoherencyEnforcement.revert_unit)
    assert all(np.allclose(m.location, s) for m, s in zip(reverted, starts)), (
        "revert_unit should send the whole unit home"
    )

    repaired = make_unit(starts, ends)
    moved = enforce_after_move(repaired, NEAREST, FURTHEST, CoherencyEnforcement.repair)

    assert coherent(repaired), "repair must leave the unit legal"
    assert moved > 0
    # The four that were never in breach keep exactly the ground they took.
    for index in range(4):
        assert np.allclose(repaired[index].location, ends[index]), (
            f"model {index} did not break the unit and should have kept its move"
        )
    # And the unit as a whole advanced, where the revert left it at its start.
    assert repaired[0].location[0] > reverted[0].location[0]


def test_repair_falls_back_to_the_revert_when_it_cannot_gather() -> None:
    """A breach repair cannot close still gets the spec's own consequence.

    A pure *spread* breach is the case: every model has a neighbour within the
    chain distance, so there is no stray to pull in, but the unit is strung out
    past the 9in cap. Repair declines and `revert_unit` finishes the job, so the
    guarantee "the board is legal after enforcement" is never weakened.

    **Six models, not five.** At the test fixture's `base_radius` of 0 a line of
    five at the 2in chain limit spans only 8in, so a pure spread breach cannot be
    built from five at all -- the two conditions cannot be separated until the
    unit is big enough for the chain to reach past the cap.
    """
    # A chain at exactly the 2in limit: every neighbour is legal, the span is not.
    starts = [(10.0 + 1.5 * i, 10.0) for i in range(6)]
    ends = [(10.0 + 2.0 * i, 10.0) for i in range(6)]
    models = make_unit(starts, ends)
    assert not coherent(models), "fixture must start illegal"

    enforce_after_move(models, NEAREST, FURTHEST, CoherencyEnforcement.repair)

    assert coherent(models)
    assert all(np.allclose(m.location, s) for m, s in zip(models, starts)), (
        "an ungatherable unit must fall back to the full revert"
    )


def test_repair_touches_nothing_when_the_move_was_already_legal() -> None:
    """No breach, no edit — the mode must be inert on a legal move."""
    starts = [(10.0 + 1.5 * i, 10.0) for i in range(5)]
    ends = [(11.0 + 1.5 * i, 10.0) for i in range(5)]
    models = make_unit(starts, ends)
    assert coherent(models), "fixture must start legal"

    moved = enforce_after_move(models, NEAREST, FURTHEST, CoherencyEnforcement.repair)

    assert moved == 0
    assert all(np.allclose(m.location, e) for m, e in zip(models, ends))
