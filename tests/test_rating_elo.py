"""Ratings are fitted, not accumulated — and the two advantages are fitted too.

Sequential K-factor updates exist because online chess cannot re-fit. Here the
pool is fixed and any match can be replayed, so a sequential rule would only
make a rating depend on the *order* games were played in -- gratuitous
irreproducibility in a project whose training is otherwise bit-reproducible.
"""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.rating.elo import (
    Design,
    bootstrap_ratings,
    fit_ratings,
    win_probability,
)
from wargame_rl.wargame.rating.score import margin_score

ENTRANTS = ("random", "squad_march", "squad_march_shoot")


def _synthetic_design(
    ratings: dict[str, float],
    zone_advantage: float,
    turn_advantage: float,
    n_layouts: int = 400,
    seed: int = 0,
    entrants: tuple[str, ...] = ENTRANTS,
    noise: float = 0.0,
    layout_noise: float = 0.0,
) -> Design:
    """Every pairing over the balanced four legs, scored from planted values.

    Deterministic scores by default: the fit is being checked, not the sampling.
    `noise` perturbs each game independently; `layout_noise` is a shock *shared*
    by every game on a layout, which is the correlation a layout bootstrap
    exists to capture and a row bootstrap cannot see.
    """
    rng = np.random.default_rng(seed)
    layout_shock = (
        rng.normal(0.0, layout_noise, size=n_layouts) if layout_noise else None
    )
    index_a: list[int] = []
    index_b: list[int] = []
    sigma_zone: list[float] = []
    sigma_turn: list[float] = []
    score: list[float] = []
    layout: list[int] = []

    for layout_index in range(n_layouts):
        for a in range(len(entrants)):
            for b in range(a + 1, len(entrants)):
                for zone in (1.0, -1.0):
                    for turn in (1.0, -1.0):
                        difference = (
                            ratings[entrants[a]]
                            - ratings[entrants[b]]
                            + zone * zone_advantage
                            + turn * turn_advantage
                        )
                        expected = 1.0 / (1.0 + 10.0 ** (-difference / 400.0))
                        if noise > 0.0 or layout_shock is not None:
                            shock = (
                                0.0
                                if layout_shock is None
                                else float(layout_shock[layout_index])
                            )
                            jitter = rng.normal(0.0, noise) if noise > 0.0 else 0.0
                            expected = float(
                                np.clip(expected + shock + jitter, 0.001, 0.999)
                            )
                        index_a.append(a)
                        index_b.append(b)
                        sigma_zone.append(zone)
                        sigma_turn.append(turn)
                        score.append(expected)
                        layout.append(layout_index)

    return Design(
        index_a=np.array(index_a, dtype=np.intp),
        index_b=np.array(index_b, dtype=np.intp),
        sigma_zone=np.array(sigma_zone, dtype=np.float64),
        sigma_turn=np.array(sigma_turn, dtype=np.float64),
        score=np.array(score, dtype=np.float64),
        layout=np.array(layout, dtype=np.intp),
    )


def test_the_fit_recovers_known_ratings() -> None:
    planted = {"random": 0.0, "squad_march": 250.0, "squad_march_shoot": 500.0}
    design = _synthetic_design(planted, zone_advantage=0.0, turn_advantage=0.0)

    fit = fit_ratings(design, ENTRANTS, anchor="random")

    assert fit.converged
    for name, value in planted.items():
        assert fit.ratings[ENTRANTS.index(name)] == pytest.approx(value, abs=6.0)


def test_the_fit_recovers_the_zone_and_turn_advantages() -> None:
    """Rather than merely cancelling an imbalance known to exist, the fit
    *reports* it -- a result in its own right, and one available before any
    training happens."""
    planted = {"random": 0.0, "squad_march": 200.0, "squad_march_shoot": 350.0}
    design = _synthetic_design(planted, zone_advantage=40.0, turn_advantage=-25.0)

    fit = fit_ratings(design, ENTRANTS, anchor="random")

    assert fit.h_zone == pytest.approx(40.0, abs=5.0)
    assert fit.h_turn == pytest.approx(-25.0, abs=5.0)


def test_the_anchor_is_pinned_at_zero() -> None:
    """Ratings are identified only up to a common shift. Anchoring on the floor
    makes "zero Elo" mean "no better than random", which is the interpretation
    this repo already uses for the `random` baseline."""
    design = _synthetic_design(
        {"random": 100.0, "squad_march": 300.0, "squad_march_shoot": 600.0}, 0.0, 0.0
    )

    fit = fit_ratings(design, ENTRANTS, anchor="random")

    assert fit.ratings[ENTRANTS.index("random")] == 0.0


def test_an_undefeated_entrant_stays_finite() -> None:
    """Without a prior, an entrant that never loses runs to infinity.
    `squad_march_shoot` has genuinely scored a 1.00 win rate on some configs
    here, so this is the ordinary case rather than a pathological one."""
    # Entrant 1 is A in every row and scores 1.0 in every row: undefeated.
    design = Design(
        index_a=np.ones(40, dtype=np.intp),
        index_b=np.zeros(40, dtype=np.intp),
        sigma_zone=np.tile([1.0, -1.0], 20),
        sigma_turn=np.tile([1.0, 1.0, -1.0, -1.0], 10),
        score=np.ones(40),
        layout=np.arange(40, dtype=np.intp) // 4,
    )

    fit = fit_ratings(design, ("random", "unbeaten"), anchor="random")

    assert np.all(np.isfinite(fit.ratings))
    assert fit.ratings[1] > 0.0


def test_swapping_zone_and_turn_together_is_refused() -> None:
    """This is *why* the schedule has four legs and not two.

    Varying the two axes together identifies only their sum. Regularising the
    pair would return a plausible split of a quantity the data never separated,
    and a table reporting a deployment-zone advantage it could not have measured
    is worse than no table -- so the fit refuses instead.
    """
    design = _synthetic_design(
        {"random": 0.0, "squad_march": 200.0, "squad_march_shoot": 350.0},
        zone_advantage=60.0,
        turn_advantage=0.0,
    )
    confounded = Design(
        index_a=design.index_a,
        index_b=design.index_b,
        sigma_zone=design.sigma_zone,
        sigma_turn=design.sigma_zone,  # the two axes never vary independently
        score=design.score,
        layout=design.layout,
    )

    with pytest.raises(ValueError, match="four legs"):
        fit_ratings(confounded, ENTRANTS, anchor="random")


def test_the_bootstrap_resamples_layouts_not_games() -> None:
    """The four legs played on one layout are **one** unit of evidence, not
    four. Resampling games would treat them as independent and report an
    interval about half the width it should be."""
    design = _synthetic_design(
        {"random": 0.0, "squad_march": 200.0, "squad_march_shoot": 350.0},
        30.0,
        10.0,
        n_layouts=40,
        noise=0.08,
    )

    table = bootstrap_ratings(design, ENTRANTS, n_bootstrap=60, seed=0)

    assert table.n_bootstrap == 60
    assert np.all(table.lower <= table.fit.ratings)
    assert np.all(table.fit.ratings <= table.upper)
    assert table.lower[ENTRANTS.index("random")] == 0.0


def test_the_layout_bootstrap_is_wider_than_resampling_rows() -> None:
    """Layouts are the resampling unit, and this is what that buys.

    `docs/elo.md` justifies the bootstrap by arguing the quasi-likelihood
    Hessian *understates* the error. That reason does not survive contact: any
    `[0, 1]`-valued score has variance at most `p(1-p)`, so the Bernoulli
    assumption bounds the marginal variance from **above** and is conservative
    on its own -- the same fact as the Rao-Blackwellisation argument that
    motivates the score in the first place.

    What the Hessian genuinely cannot see is **dependence between rows**: the
    four legs on one layout share terrain, objectives and dice. Here the same
    games are bootstrapped twice, once grouped by layout and once with every
    game its own layout, and only the grouping differs -- the layout bootstrap
    must come out wider or the grouping bought nothing.
    """
    design = _synthetic_design(
        {"random": 0.0, "squad_march": 200.0, "squad_march_shoot": 350.0},
        30.0,
        10.0,
        n_layouts=40,
        noise=0.10,
        layout_noise=0.16,
    )
    ungrouped = Design(
        index_a=design.index_a,
        index_b=design.index_b,
        sigma_zone=design.sigma_zone,
        sigma_turn=design.sigma_turn,
        score=design.score,
        layout=np.arange(design.n_games, dtype=np.intp),
    )

    by_layout = bootstrap_ratings(design, ENTRANTS, n_bootstrap=200, seed=1)
    by_row = bootstrap_ratings(ungrouped, ENTRANTS, n_bootstrap=200, seed=1)
    index = ENTRANTS.index("squad_march_shoot")

    assert (by_layout.upper[index] - by_layout.lower[index]) > (
        by_row.upper[index] - by_row.lower[index]
    )


def test_the_bootstrap_reproduces_from_its_seed() -> None:
    design = _synthetic_design(
        {"random": 0.0, "squad_march": 200.0, "squad_march_shoot": 350.0},
        20.0,
        0.0,
        n_layouts=30,
        noise=0.05,
    )

    first = bootstrap_ratings(design, ENTRANTS, n_bootstrap=40, seed=7)
    second = bootstrap_ratings(design, ENTRANTS, n_bootstrap=40, seed=7)

    np.testing.assert_array_equal(first.lower, second.lower)
    np.testing.assert_array_equal(first.upper, second.upper)


def test_delta_elo_matches_the_linear_approximation() -> None:
    """Near equality, `dR ~ 173.7 * mean_margin / s_m`. This is the sanity
    bridge between a rating and a `measure-paired` number: if the two disagree
    badly, one of them is measuring a different scenario."""
    margin_scale = 50.0
    mean_margin = 10.0
    scores = margin_score(np.full(4000, mean_margin), margin_scale)
    design = Design(
        index_a=np.zeros(scores.size, dtype=np.intp),
        index_b=np.ones(scores.size, dtype=np.intp),
        sigma_zone=np.tile([1.0, -1.0], scores.size // 2),
        sigma_turn=np.tile([1.0, 1.0, -1.0, -1.0], scores.size // 4),
        score=scores,
        layout=np.arange(scores.size, dtype=np.intp) // 4,
    )

    fit = fit_ratings(design, ("b", "a"), anchor="a")
    predicted = 400.0 / np.log(10.0) * mean_margin / margin_scale

    assert fit.ratings[0] == pytest.approx(predicted, rel=0.05)


def test_win_probability_is_the_elo_curve() -> None:
    """`p` feeds PFSP opponent sampling, so the two subsystems compose rather
    than maintaining two different win-rate estimates."""
    assert win_probability(1500.0, 1500.0) == pytest.approx(0.5)
    assert win_probability(1900.0, 1500.0) == pytest.approx(10 / 11, rel=1e-6)


def test_an_unknown_anchor_is_refused() -> None:
    design = _synthetic_design(
        {"random": 0.0, "squad_march": 100.0, "squad_march_shoot": 200.0}, 0.0, 0.0
    )

    with pytest.raises(ValueError, match="anchor"):
        fit_ratings(design, ENTRANTS, anchor="not_an_entrant")


def test_a_self_pairing_measures_the_seat_advantage_alone() -> None:
    """An entrant against itself is the cleanest estimator of the two
    advantages there is.

    The rating difference is identically zero by construction, so whatever
    margin survives the balanced four legs is the seat advantage and nothing
    else. This is what the seat-parity gate leans on -- and it only works
    because the design matrix accumulates a self-pairing to zero rather than
    leaving a stray `+1`.
    """
    n_layouts = 200
    rows = []
    for layout_index in range(n_layouts):
        for zone in (1.0, -1.0):
            for turn in (1.0, -1.0):
                difference = zone * 45.0 + turn * 15.0
                rows.append(
                    (
                        zone,
                        turn,
                        1.0 / (1.0 + 10.0 ** (-difference / 400.0)),
                        layout_index,
                    )
                )
    zones, turns, scores, layouts = (np.array(column) for column in zip(*rows))
    design = Design(
        index_a=np.ones(len(rows), dtype=np.intp),
        index_b=np.ones(len(rows), dtype=np.intp),
        sigma_zone=zones,
        sigma_turn=turns,
        score=scores,
        layout=layouts.astype(np.intp),
    )

    fit = fit_ratings(design, ("random", "mirror"), anchor="random")

    assert fit.h_zone == pytest.approx(45.0, abs=3.0)
    assert fit.h_turn == pytest.approx(15.0, abs=3.0)
    assert fit.ratings[1] == pytest.approx(0.0, abs=1.0)
