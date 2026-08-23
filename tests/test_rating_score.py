"""The victory-point margin becomes the score a rating is fitted on.

Win rate cannot resolve differences under ~7pp on these configs while
`vp_margin` separates cleanly -- TF32 is the reference case, costing 8.5
vp_margin on both seeds while win rate moved 0.705 -> 0.65, inside the noise. A
rating built on win/draw/loss inherits that blindness exactly, so the margin has
to enter the rating, and it enters through the *score* rather than through the
update rule.
"""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.rating.score import (
    DEFAULT_MARGIN_SCALE,
    fit_margin_scale,
    margin_score,
)


def test_a_tie_is_exactly_a_draw() -> None:
    """`s(0) = 1/2` with no special case, which is why a VP tie needs none."""
    assert margin_score(np.array([0.0]))[0] == 0.5


def test_the_score_saturates() -> None:
    """Per-episode `vp_margin` sd is 45-50 on 25v25, so blowouts are common. An
    unbounded score would let one layout dominate a whole rating."""
    scores = margin_score(np.array([-1e6, -200.0, 200.0, 1e6]))

    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)
    assert np.all(np.isfinite(scores))


def test_a_huge_negative_margin_does_not_overflow() -> None:
    """The naive `1/(1+exp(-m/s))` overflows here and returns a warning plus a
    nan; a rating is not allowed to depend on which sign the blowout had."""
    with np.errstate(over="raise"):
        assert margin_score(np.array([-1e8]), 1.0)[0] == pytest.approx(0.0)


def test_the_score_is_monotonic_in_the_margin() -> None:
    margins = np.array([-100.0, -10.0, 0.0, 10.0, 100.0])

    assert np.all(np.diff(margin_score(margins)) > 0)


def test_a_zero_scale_degrades_to_the_win_indicator() -> None:
    """Setting the scale to zero must recover plain Elo. This is the check that
    the margin score *generalises* the win/draw/loss outcome rather than
    replacing it."""
    scores = margin_score(np.array([-5.0, 0.0, 5.0]), 0.0)

    assert scores.tolist() == [0.0, 0.5, 1.0]


def test_the_scale_is_recovered_from_data_generated_with_it() -> None:
    """`s_m` is fitted so `s ~ P(win | margin m)`: the noisy 0/1 outcome is
    replaced by its own conditional expectation, which is why a rating point
    still means a win probability."""
    rng = np.random.default_rng(0)
    planted = 37.0
    margins = rng.normal(0.0, 60.0, size=20_000)
    wins = (rng.random(margins.size) < margin_score(margins, planted)).astype(float)

    assert fit_margin_scale(margins, wins) == pytest.approx(planted, rel=0.08)


def test_the_pinned_default_is_in_the_range_the_scenario_produces() -> None:
    """Pinned rather than fitted per run, so a rating is reproducible. It must
    still be re-fitted when the scenario changes -- on a config where VP is
    capped the margin distribution is bounded, and a scale fitted on an uncapped
    scenario would be wrong in a direction nobody would notice."""
    assert 10.0 < DEFAULT_MARGIN_SCALE < 200.0


def test_fitting_refuses_a_degenerate_corpus() -> None:
    """All-wins carries no information about the scale, and silently returning
    something would put a fabricated number into every rating downstream."""
    margins = np.array([10.0, 20.0, 30.0])

    with pytest.raises(ValueError, match="both"):
        fit_margin_scale(margins, np.ones(3))
