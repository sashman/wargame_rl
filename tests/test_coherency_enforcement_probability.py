"""Partial coherency enforcement: `enforce_move_probability`.

Full enforcement is not free. Warm-started from a trained policy it reaches a
coherency rate of 1.000 with zero models adrift, and `vp_margin` settles at
~60-69 against the unenforced control's ~92.5 — flat over a hundred epochs, so
not a transient. Training into it from scratch is worse (-75.3). The probability
turns that cliff into a dial.

The load-bearing test is `test_full_enforcement_is_unchanged_by_this_parameter`:
the default must take the same code path as before the parameter existed, or
every measured baseline moves for a reason nobody chose.
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


def broken_unit() -> list[WargameModel]:
    """Three models whose move left the unit illegal: two together, one 23 away.

    That distance breaches the *spread* cap as well as the chain, and spread is
    collective — once one model is beyond it, no model is within it of every
    other, so every model is in breach and `revert_model` escalates to the whole
    unit. The count is therefore 3, not 1. What these tests care about is
    whether the revert fires at all, not how wide it is.
    """
    starts = [(0.0, 0.0), (1.5, 0.0), (3.0, 0.0)]
    ends = [(0.0, 0.0), (1.5, 0.0), (23.0, 0.0)]
    models = []
    for start, end in zip(starts, ends):
        model = WargameModel(
            location=position(*end),
            distances_to_objectives=np.zeros((1, 2)),
            stats={"current_wounds": 1, "max_wounds": 1},
            group_id=0,
        )
        model.previous_location = np.array(start, dtype=model.location.dtype)
        models.append(model)
    return models


def test_full_enforcement_is_unchanged_by_this_parameter() -> None:
    # Arrange: the same illegal move, enforced the old way and the new default.
    with_default = broken_unit()
    explicit_one = broken_unit()

    # Act
    a = enforce_after_move(
        with_default, NEAREST, FURTHEST, CoherencyEnforcement.revert_model
    )
    b = enforce_after_move(
        explicit_one,
        NEAREST,
        FURTHEST,
        CoherencyEnforcement.revert_model,
        probability=1.0,
        rng=np.random.default_rng(0),
    )

    # Assert: identical, and no draw was needed for the default.
    assert a == b == 3
    assert [tuple(m.location) for m in with_default] == [
        tuple(m.location) for m in explicit_one
    ]


def test_zero_probability_matches_enforcement_off() -> None:
    # Arrange
    at_zero = broken_unit()
    switched_off = broken_unit()

    # Act
    reverted = enforce_after_move(
        at_zero,
        NEAREST,
        FURTHEST,
        CoherencyEnforcement.revert_model,
        probability=0.0,
        rng=np.random.default_rng(0),
    )
    off = enforce_after_move(switched_off, NEAREST, FURTHEST, CoherencyEnforcement.off)

    # Assert: the illegal move stands in both cases.
    assert reverted == off == 0
    assert at_zero[2].location.tolist() == [23.0, 0.0]


def test_a_partial_probability_reverts_roughly_that_fraction() -> None:
    # Arrange: one broken unit per trial, so each trial is one draw.
    rng = np.random.default_rng(11)
    trials = 400

    # Act
    reverted = sum(
        enforce_after_move(
            broken_unit(),
            NEAREST,
            FURTHEST,
            CoherencyEnforcement.revert_model,
            probability=0.5,
            rng=rng,
        )
        > 0
        for _ in range(trials)
    )

    # Assert: binomial(400, 0.5) has sd 10, so 5 sd is a safe envelope while
    # still failing on a parameter that is ignored (400) or inverted (0).
    assert 150 < reverted < 250


def test_below_one_without_an_rng_is_refused() -> None:
    # Arrange / Act / Assert: silently falling back to an unseeded draw would
    # make a seeded run irreproducible, which every golden gate here relies on.
    with pytest.raises(ValueError, match="rng"):
        enforce_after_move(
            broken_unit(),
            NEAREST,
            FURTHEST,
            CoherencyEnforcement.revert_model,
            probability=0.5,
        )


@pytest.mark.parametrize("probability", [0.25, 0.5, 0.75])
def test_the_same_seed_reproduces_the_same_reverts(probability: float) -> None:
    # Arrange: a seeded run must reproduce, or partial enforcement cannot be
    # compared across arms at all.
    def run() -> list[int]:
        rng = np.random.default_rng(7)
        return [
            enforce_after_move(
                broken_unit(),
                NEAREST,
                FURTHEST,
                CoherencyEnforcement.revert_model,
                probability=probability,
                rng=rng,
            )
            for _ in range(40)
        ]

    # Act / Assert
    assert run() == run()
