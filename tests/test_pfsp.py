"""Prioritised opponent sampling: the weights, and the cases that break them.

Pure numpy, so these run with no env, no torch and no GPU -- the same property
`test_rating_elo.py` relies on.
"""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.rating.pfsp import pfsp_weights, sample_opponent


def test_hard_mode_prefers_the_opponents_the_learner_loses_to() -> None:
    weights = pfsp_weights(np.array([0.9, 0.5, 0.1]), mode="hard")

    assert weights[2] > weights[1] > weights[0]
    assert weights.sum() == pytest.approx(1.0)


def test_even_mode_prefers_the_level_matchups() -> None:
    """Not the same ordering as `hard`, which is the point of having both.

    A 0.1 opponent is the *hardest* and the *least* even, so a mode that ranked
    them the same way would be `hard` under another name.
    """
    weights = pfsp_weights(np.array([0.9, 0.5, 0.1]), mode="even")

    assert weights[1] > weights[0]
    assert weights[1] > weights[2]
    assert weights[0] == pytest.approx(weights[2])


def test_uniform_is_the_control() -> None:
    """A pool changes training on its own; the schedule is a separate claim.

    Without this mode an arm comparing PFSP against no self-play at all would
    confound the two, which is the measurement error this repo makes most often.
    """
    weights = pfsp_weights(np.array([0.9, 0.5, 0.1]), mode="uniform")

    assert np.allclose(weights, 1.0 / 3.0)


def test_no_snapshot_is_starved() -> None:
    """An opponent that is never sampled cannot catch the learner forgetting it,
    and forgetting is the failure a pool exists to prevent."""
    weights = pfsp_weights(np.array([1.0, 1.0, 0.0]), mode="hard", uniform_floor=0.1)

    assert float(weights.min()) > 0.0
    assert float(weights.min()) == pytest.approx(0.1 / 3.0)


def test_a_completely_beaten_pool_falls_back_to_uniform() -> None:
    """The state a pool reaches when the learner has outgrown everything in it.

    Every weight is zero there, so normalising would divide by zero. The right
    response is to keep playing until a new snapshot is taken, not to crash.
    """
    weights = pfsp_weights(np.ones(4), mode="hard")

    assert np.allclose(weights, 0.25)


def test_a_zero_floor_is_pure_prioritisation() -> None:
    weights = pfsp_weights(np.array([0.5, 0.0]), mode="hard", uniform_floor=0.0)

    assert weights[0] == pytest.approx(0.2)
    assert weights[1] == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("probability", "floor", "mode"),
    [
        (np.array([1.5]), 0.1, "hard"),
        (np.array([-0.1]), 0.1, "hard"),
        (np.array([0.5]), 1.5, "hard"),
        (np.array([]), 0.1, "hard"),
        (np.array([0.5]), 0.1, "greedy"),
    ],
)
def test_bad_input_raises(probability: np.ndarray, floor: float, mode: str) -> None:
    """An out-of-range `p` means the caller read the rating table wrong, and a
    silently clipped weight would hide that behind a plausible schedule."""
    with pytest.raises(ValueError):
        pfsp_weights(probability, mode=mode, uniform_floor=floor)  # type: ignore[arg-type]


def test_sampling_is_reproducible_from_its_generator() -> None:
    """Training is deterministic given seed, config and code, and an opponent
    schedule drawn from a shared or wall-clock-seeded source would end that."""
    weights = pfsp_weights(np.array([0.9, 0.5, 0.1]), mode="hard")

    first = [sample_opponent(weights, np.random.default_rng(7)) for _ in range(5)]
    again = [sample_opponent(weights, np.random.default_rng(7)) for _ in range(5)]

    assert first == again


def test_sampling_follows_the_weights() -> None:
    weights = pfsp_weights(np.array([0.99, 0.01]), mode="hard", uniform_floor=0.0)
    rng = np.random.default_rng(0)

    draws = [sample_opponent(weights, rng) for _ in range(2000)]

    assert sum(draws) > 1900
