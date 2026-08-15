"""`CoherencyEnforcement.clamp`: shorten the move instead of cancelling it.

The revert modes were measured to cost movement, not just legality -- on the
real tables `revert_model` sends back 6.34 of 25 models every step and cuts mean
displacement 20%. Clamping keeps the model's chosen direction and stops it at
the legal distance.

Two tests carry the weight. `test_a_clamped_model_ends_up_coherent` is the
guarantee: a mode that shortened a move without making it legal would be strictly
worse than reverting, since it would neither enforce the rule nor preserve the
policy's decision. `test_the_clamp_falls_back_to_a_revert_when_it_cannot_help`
covers the case the mechanism cannot fix at all -- a unit already broken when its
move began, which no shortening repairs.
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


# The shipped 32mm infantry base, and the value these tests must run at. Every
# test in this file originally ran at the `WargameModel` default of 0.0, where
# `_overlaps_any_base` and `_cascade_displaced` are *unconditional no-ops* -- so
# the whole overlap half of the mode was untested, and a real bug shipped behind
# a green suite. The revert modes have had 0.63 coverage all along; clamp
# inherited none of it. (A docstring here rather than a comment trips
# `check-docstring-first`, which is how this file failed its first commit.)
BASE_RADIUS = 0.63


def unit_from(
    starts: list[tuple[float, float]],
    ends: list[tuple[float, float]],
    group_id: int = 0,
) -> list[WargameModel]:
    """One unit whose models moved from `starts` to `ends`, on real bases."""
    models = []
    for start, end in zip(starts, ends):
        model = WargameModel(
            location=position(*end),
            distances_to_objectives=np.zeros((1, 2)),
            stats={"current_wounds": 1, "max_wounds": 1},
            group_id=group_id,
        )
        model.base_radius = BASE_RADIUS
        model.previous_location = np.array(start, dtype=model.location.dtype)
        models.append(model)
    return models


def is_coherent(models: list[WargameModel]) -> bool:
    """Whether every unit satisfies the rule at the models' current positions."""
    report = evaluate_coherency(
        positions=np.array([m.location for m in models], dtype=float),
        group_ids=np.array([m.group_id for m in models], dtype=np.intp),
        alive_mask=np.array([m.is_alive for m in models], dtype=bool),
        base_radii=np.array([m.base_radius for m in models], dtype=float),
        nearest_distance=NEAREST,
        furthest_distance=FURTHEST,
    )
    return all(unit.coherent for unit in report.units)


def a_unit_that_ran_away() -> list[WargameModel]:
    """Two models hold station; the third sprints 12 out along +x.

    12 breaks the chain (2) but the *body* stays legal, which is the case
    clamping exists for and the case that dominates the measured defect: 82% of
    adrift models are walking to a different objective, a median 13.6 out.
    """
    return unit_from(
        [(0.0, 0.0), (1.5, 0.0), (3.0, 0.0)],
        [(0.0, 0.0), (1.5, 0.0), (12.0, 0.0)],
    )


def test_a_clamped_model_ends_up_coherent() -> None:
    # Arrange: the guarantee. Shortening a move without making it legal would
    # be worse than reverting -- neither enforced nor faithful to the policy.
    models = a_unit_that_ran_away()
    assert not is_coherent(models), "test is vacuous unless the move was illegal"

    # Act
    moved = enforce_after_move(models, NEAREST, FURTHEST, CoherencyEnforcement.clamp)

    # Assert
    assert moved == 1
    assert is_coherent(models)


def test_the_clamped_model_keeps_its_direction_and_most_of_its_ground() -> None:
    # Arrange: the whole point. A revert sends this model back to 3.0; a clamp
    # should leave it as far out as the rule allows, on the same line.
    models = a_unit_that_ran_away()

    # Act
    enforce_after_move(models, NEAREST, FURTHEST, CoherencyEnforcement.clamp)

    # Assert: still on the +x axis, ahead of where it started, and short of
    # where it asked to go.
    x, y = float(models[2].location[0]), float(models[2].location[1])
    assert y == pytest.approx(0.0)
    assert 3.0 < x < 12.0
    # Base to base, like every other distance here: the gap between the discs
    # is NEAREST, so the centres sit NEAREST + two radii apart.
    assert x == pytest.approx(1.5 + NEAREST + 2 * BASE_RADIUS)


def test_clamping_beats_reverting_on_ground_kept() -> None:
    # Arrange: the same illegal move under both modes. This is the comparison
    # the mode exists to win, so it is asserted rather than assumed.
    clamped = a_unit_that_ran_away()
    reverted = a_unit_that_ran_away()

    # Act
    enforce_after_move(clamped, NEAREST, FURTHEST, CoherencyEnforcement.clamp)
    enforce_after_move(reverted, NEAREST, FURTHEST, CoherencyEnforcement.revert_unit)

    # Assert: both legal, and the clamp gives up less ground.
    assert is_coherent(clamped) and is_coherent(reverted)
    assert float(clamped[2].location[0]) > float(reverted[2].location[0])


def test_a_legal_move_is_untouched() -> None:
    # Arrange: a unit that stayed in coherency must not be nudged at all --
    # enforcement that fires on legal play is a movement tax by another name.
    models = unit_from(
        [(0.0, 0.0), (1.5, 0.0), (3.0, 0.0)],
        [(0.5, 0.0), (2.0, 0.0), (3.5, 0.0)],
    )

    # Act
    moved = enforce_after_move(models, NEAREST, FURTHEST, CoherencyEnforcement.clamp)

    # Assert
    assert moved == 0
    assert [tuple(m.location) for m in models] == [(0.5, 0.0), (2.0, 0.0), (3.5, 0.0)]


def test_the_clamp_falls_back_to_a_revert_when_it_cannot_help() -> None:
    # Arrange: the unit was ALREADY broken when the move began -- the two halves
    # start 30 apart, which no shortening of anyone's move repairs. The mode
    # must not silently leave an illegal position.
    models = unit_from(
        [(0.0, 0.0), (1.5, 0.0), (30.0, 0.0)],
        [(0.5, 0.0), (2.0, 0.0), (31.0, 0.0)],
    )

    # Act
    moved = enforce_after_move(models, NEAREST, FURTHEST, CoherencyEnforcement.clamp)

    # Assert: everyone is back at their start, exactly as `revert_unit` would.
    assert moved == 3
    assert [tuple(m.location) for m in models] == [
        (0.0, 0.0),
        (1.5, 0.0),
        (30.0, 0.0),
    ]


def test_a_clamped_model_never_leaves_its_own_move_segment() -> None:
    # Arrange: the invariant a first draft of this mode broke. Pulling straight
    # at the body moved a model 27 units *backwards* -- further than its speed
    # allows, and enough to repair a break its move never caused. The clamp is
    # only ever a shortening, so the result must lie on the segment from where
    # the model started to where it asked to go, and never past the start.
    models = unit_from(
        [(0.0, 0.0), (1.5, 0.0), (3.0, 1.0)],
        [(0.0, 0.0), (1.5, 0.0), (9.0, 7.0)],
    )
    start = np.array([3.0, 1.0])
    destination = np.array([9.0, 7.0])

    # Act
    enforce_after_move(models, NEAREST, FURTHEST, CoherencyEnforcement.clamp)

    # Assert: on the segment, and no further back than the start.
    landed = np.array(models[2].location, dtype=float)
    travelled = landed - start
    intended = destination - start
    fraction = float(travelled @ intended) / float(intended @ intended)
    assert 0.0 <= fraction <= 1.0
    assert np.allclose(landed, start + intended * fraction, atol=1e-6)


def test_units_are_clamped_independently() -> None:
    # Arrange: one broken unit and one intact one, far apart. A mode that
    # evaluated the force as a whole would drag the legal unit around too.
    models = unit_from(
        [(0.0, 0.0), (1.5, 0.0), (40.0, 0.0), (41.5, 0.0)],
        [(0.0, 0.0), (12.0, 0.0), (40.5, 0.0), (42.0, 0.0)],
    )
    for model in models[2:]:
        model.group_id = 1

    # Act
    enforce_after_move(models, NEAREST, FURTHEST, CoherencyEnforcement.clamp)

    # Assert
    assert is_coherent(models)
    assert tuple(models[2].location) == (40.5, 0.0)
    assert tuple(models[3].location) == (42.0, 0.0)


def test_a_fallback_revert_never_leaves_two_bases_overlapping() -> None:
    # Arrange: the guarantee `clamp` claims -- "exactly `revert_unit`'s" -- and
    # the one it did not keep. Unit 0 was ALREADY broken when it moved, so it
    # cannot be clamped and falls back to a full revert; unit 1's model legally
    # advanced onto the ground unit 0's model is about to be sent back to.
    # Reverting without cascading drops one base on top of another, which
    # `03-moving.md` forbids in the same breath as coherency.
    #
    # Invisible at `base_radius: 0.0`, where the overlap checks are no-ops --
    # which is exactly how it shipped.
    broken = unit_from(
        [(0.0, 0.0), (1.5, 0.0), (40.0, 0.0)],
        [(0.5, 0.0), (2.0, 0.0), (40.5, 0.0)],
    )
    # The taker must stay COHERENT after its move, or it falls back too and
    # vacates the ground -- which is what made the first draft of this test
    # pass while the bug was present.
    taker = unit_from([(8.0, 0.0), (9.4, 0.0)], [(0.0, 0.0), (1.4, 0.0)], group_id=1)
    models = broken + taker

    # Act
    enforce_after_move(models, NEAREST, FURTHEST, CoherencyEnforcement.clamp)

    # Assert: no two live bases overlap anywhere on the board.
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            gap = float(np.hypot(*(np.array(a.location) - np.array(b.location))))
            assert gap >= a.base_radius + b.base_radius - 1e-9, (
                f"bases overlap: gap {gap:.3f} < {a.base_radius + b.base_radius:.3f}"
            )


def test_the_other_modes_are_unchanged_by_this_addition() -> None:
    # Arrange: `clamp` is a new branch taken before the revert machinery, so the
    # existing modes must reach it not at all. Both golden gates cover the
    # default; this pins the two named modes directly.
    for mode in (CoherencyEnforcement.revert_unit, CoherencyEnforcement.revert_model):
        models = a_unit_that_ran_away()

        # Act
        enforce_after_move(models, NEAREST, FURTHEST, mode)

        # Assert: the runaway is back where it started, not clamped.
        assert tuple(models[2].location) == (3.0, 0.0)
