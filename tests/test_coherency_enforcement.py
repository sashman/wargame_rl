"""End-of-move coherency enforcement: the two reverts, and the no-op default.

`03-moving.md` § Making a move undoes a move that ends a unit out of coherency.
These pin both modes against hand-placed geometry, so the revert is checked on
positions rather than on whether a policy happened to break formation.
"""

from __future__ import annotations

import numpy as np
import pytest

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
    # spread cap -- 5 from the nearest, 6.5 from the furthest, both under 9.
    # That isolation matters: a straggler far enough to breach spread as well
    # puts every model in the unit in breach, and then the two modes coincide.
    starts = [(0.0, 0.0), (1.5, 0.0), (3.0, 0.0)]
    ends = [(20.0, 0.0), (21.5, 0.0), (15.0, 0.0)]
    models = make_unit(starts, ends)

    # Act
    reverted = enforce_after_move(
        models, NEAREST, FURTHEST, CoherencyEnforcement.revert_model
    )

    # Assert: the straggler goes back, the pair keeps the ground it took.
    assert reverted == 1
    assert locations(models)[:2] == [(20.0, 0.0), (21.5, 0.0)]
    assert models[2].location.tolist() == [3.0, 0.0]


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
    # differ by 51 vp on `split_evenly`.
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
