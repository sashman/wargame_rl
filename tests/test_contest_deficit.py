"""`contest_deficit` widens which objectives the travel reward may point at.

The gate inside `closest_objective_v2` asks whether an arrival improves the
control label. It has always imagined exactly ONE model arriving, which means an
objective the opponent holds by two or more can never be a candidate: one body
cannot take `opponent` to `contested`. Measured on the v3.0 lineage that
accounted for 43.4% of every objective the reward could not point at, so the
agent was never paid to attack held ground.

These tests pin the three things that matter: the default reproduces the old
gate exactly, raising it opens up exactly the objectives that were excluded, and
the parameter is validated at construction rather than at runtime.
"""

from __future__ import annotations

import pytest

from wargame_rl.wargame.envs.reward.calculators.closest_objective_v2 import (
    ClosestObjectiveV2Calculator,
)


def test_default_is_the_historical_one_model_gate() -> None:
    """At the default, a point the opponent holds by 2+ is not a candidate."""
    # Arrange: opponent holds it 3-0, so one arrival leaves them ahead 3-1.
    calculator = ClosestObjectiveV2Calculator()

    # Act
    positive = calculator._is_positive_transition(player_count=0, opponent_count=3)

    # Assert
    assert positive is False


def test_raising_it_opens_up_ground_the_opponent_holds_by_more_than_one() -> None:
    """A unit's worth of arrivals can flip a 3-0, so it becomes a candidate."""
    # Arrange: three arrivals make it 3-3, which is `contested`, an improvement.
    calculator = ClosestObjectiveV2Calculator(contest_deficit=3)

    # Act
    positive = calculator._is_positive_transition(player_count=0, opponent_count=3)

    # Assert
    assert positive is True


@pytest.mark.parametrize(
    ("player_count", "opponent_count", "expected"),
    [
        # Unchanged by the widening: we already hold it, so no arrival improves
        # the label. This is the 56.6% of exclusions that SHOULD stay excluded.
        (2, 0, False),
        (5, 1, False),
        # Neutral and contested were already candidates at deficit 1.
        (0, 0, True),
        (2, 2, True),
    ],
)
def test_widening_does_not_pay_us_to_reinforce_what_we_already_hold(
    player_count: int, opponent_count: int, expected: bool
) -> None:
    """Only opponent-held ground changes; held and neutral points are untouched."""
    # Arrange
    wide = ClosestObjectiveV2Calculator(contest_deficit=5)

    # Act / Assert
    assert wide._is_positive_transition(player_count, opponent_count) is expected


@pytest.mark.parametrize("deficit", [0, -1])
def test_invalid_deficit_is_rejected_at_construction(deficit: int) -> None:
    """Validate at construction; `env.step()` is the hot path."""
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="contest_deficit must be >= 1"):
        ClosestObjectiveV2Calculator(contest_deficit=deficit)


def test_deficit_one_matches_deficit_default_across_the_whole_count_grid() -> None:
    """An explicit 1 and the default agree everywhere -- no silent drift."""
    # Arrange
    default = ClosestObjectiveV2Calculator()
    explicit = ClosestObjectiveV2Calculator(contest_deficit=1)

    # Act / Assert
    for player_count in range(8):
        for opponent_count in range(8):
            assert default._is_positive_transition(
                player_count, opponent_count
            ) == explicit._is_positive_transition(player_count, opponent_count)
