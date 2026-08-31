"""Fit ratings, and the two structural advantages, by maximum likelihood.

Ratings here are **fitted, not accumulated**. Sequential `K`-factor updates
exist because online chess cannot re-fit; here the pool is fixed and any match
can be replayed, so a sequential rule would only make a rating depend on the
*order* games were played in -- gratuitous irreproducibility in a project whose
training is otherwise bit-reproducible.

The model is Bradley-Terry with three explicit advantage terms::

    E[s] = 1 / (1 + 10^(-(R_A - R_B + sz*h_zone + st*h_turn + h_seat) / 400))

      sz = +1 if A deployed in zone 1, -1 if zone 2
      st = +1 if A moved first,        -1 otherwise

`h_zone`, `h_turn` and `h_seat` are the deployment-zone, first-turn and
**player-seat** advantages in Elo points, shared across all pairs. Rather than
merely cancelling an imbalance known to exist, the fit *reports* it -- a result
in its own right, available before any self-play training happens. Zone and
first turn are identified because the schedule varies those two axes
**independently**; swapping them together would confound them into one number
from which neither is recoverable, and `_require_identifiable` refuses that.

`h_seat` is different, and the difference matters when reading it. The arena
always seats entrant A on the player seat, so its column is **constant at +1**
-- there is no balancing to be had. It is identified anyway, two ways:

- **Through the pairing graph.** A constant column lies in the span of the
  `e_A - e_B` columns only if some rating vector has every pairwise difference
  equal to one, which any cycle forbids: around a triangle the differences sum
  to zero and the ones do not. So three entrants suffice, two do not.
- **Directly, by a self-pairing.** One entrant on both seats has a rating
  difference of identically zero, so whatever survives its balanced four legs is
  the seat advantage and nothing else. That is exactly what
  `just measure-seat-parity` plays.

The two differ in power, and by a lot. Measured on the exact design this
schedule produces, `se(h_seat)` against `se(h_turn)`:

    entrants  no self-pairing   one self-pairing
       2      not identified          1.41
       3           3.00               1.73
       4           2.45               1.87

So **`h_seat` is the weakest column in any table without a self-pairing**, and
feeding the seat-parity legs into the ledger is worth more than adding an
entrant. It does not replace the gate: `measure-seat-parity` reports the same
quantity in victory points, at higher power, and without assuming the advantage
is constant in Elo across pairs -- an assumption that is live, since `h_turn`
was measured to change sign with shooting.

Nothing here imports from `wargame_rl`, and scipy is deliberately not a
dependency of this project -- this is a well-behaved convex fit in a few dozen
lines of numpy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Elo's own constant: 400 points is a factor of ten in odds.
ELO_SCALE = 400.0
_LOG_TEN = float(np.log(10.0))

# Weak Gaussian prior on the ratings. Without it an undefeated entrant runs to
# infinity, and `squad_march_shoot` has genuinely scored a 1.00 win rate on some
# configs here. Also keeps the Hessian positive definite when a pairing is
# perfectly separated.
DEFAULT_PRIOR_SIGMA = 400.0


@dataclass(frozen=True, slots=True)
class Design:
    """One row per game: pairing x leg x layout.

    `layout` is the **bootstrap resampling unit** -- the four legs played on one
    layout are one piece of evidence, not four.
    """

    index_a: NDArray[np.intp]
    index_b: NDArray[np.intp]
    sigma_zone: NDArray[np.float64]
    sigma_turn: NDArray[np.float64]
    score: NDArray[np.float64]
    layout: NDArray[np.intp]

    def __post_init__(self) -> None:
        sizes = {
            self.index_a.size,
            self.index_b.size,
            self.sigma_zone.size,
            self.sigma_turn.size,
            self.score.size,
            self.layout.size,
        }
        if len(sizes) != 1:
            raise ValueError(f"every Design column must have one length, got {sizes}")

    @property
    def n_games(self) -> int:
        """How many games back the fit."""
        return int(self.score.size)


@dataclass(frozen=True, slots=True)
class RatingFit:
    """Point estimates. Quote them with an interval, never on their own."""

    entrants: tuple[str, ...]
    ratings: NDArray[np.float64]
    h_zone: float
    h_turn: float
    h_seat: float
    standard_errors: NDArray[np.float64]
    n_games: int
    n_iterations: int
    converged: bool


@dataclass(frozen=True, slots=True)
class RatingTable:
    """A fit plus its bootstrap intervals -- the thing that may be published."""

    fit: RatingFit
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    h_zone_interval: tuple[float, float]
    h_turn_interval: tuple[float, float]
    h_seat_interval: tuple[float, float]
    n_bootstrap: int
    alpha: float


def win_probability(rating_a: float, rating_b: float, advantage: float = 0.0) -> float:
    """A's expected score against B, on the Elo curve.

    This is what PFSP opponent sampling reads, so the rating table and the
    self-play sampler compose instead of maintaining two different win-rate
    estimates.

    `advantage` is whatever structural terms apply to the seating actually being
    played, in Elo points -- `h_seat` when A really is on the player seat, plus
    `h_turn` if the turn order is pinned. It defaults to zero, which is the
    **seat-neutral** comparison: "which policy is better", not "who wins this
    particular arrangement". Self-play wants the first for ranking a pool and
    the second for predicting a rollout, and they are not the same number.
    """
    difference = rating_a - rating_b + advantage
    return float(1.0 / (1.0 + 10.0 ** (-difference / ELO_SCALE)))


def rating_from_score(
    score: float,
    opponent_rating: float = 0.0,
    advantage: float = 0.0,
    max_points: float = 3.0 * ELO_SCALE,
) -> float:
    """The rating whose expected score against `opponent_rating` is `score`.

    The exact inverse of `win_probability`, so the two compose. This is how a
    measured score becomes a rating **without a fit** -- one opponent, no design
    matrix, no identifiability question -- which is all an in-run readout can
    afford between epochs.

    ⚠ **It is not a Bradley-Terry rating and must not be quoted as one.** There
    is no `h_seat`, no `h_turn`, no `h_zone` and no bootstrap, so it inherits
    every confound of the single pairing it was measured on, and it is
    comparable across runs only where the opponent and the margin scale are the
    same. `fit_ratings` is what produces a publishable number.

    `max_points` clamps the saturated ends: a score of exactly 0 or 1 inverts to
    infinity, and a mean over a handful of blowouts reaches it in float long
    before a policy is infinitely good. Three times the Elo scale is 1200
    points, past which the difference no longer carries information anyone reads.
    """
    if max_points <= 0.0:
        raise ValueError(f"max_points must be positive, got {max_points}")
    if score <= 0.0:
        difference = -max_points
    elif score >= 1.0:
        difference = max_points
    else:
        difference = ELO_SCALE * math.log10(score / (1.0 - score))
        difference = max(-max_points, min(max_points, difference))
    return float(opponent_rating - advantage + difference)


def fit_ratings(
    design: Design,
    entrants: Sequence[str],
    anchor: str = "random",
    prior_sigma: float = DEFAULT_PRIOR_SIGMA,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> RatingFit:
    """Maximise the cross-entropy over ratings, `h_zone`, `h_turn` and `h_seat`.

    Two regularisers, both necessary:

    - **One anchor pinned at zero.** Ratings are identified only up to a common
      shift, and anchoring on the floor makes "zero Elo" mean "no better than
      random" -- the interpretation this repo already uses.
    - **A weak Gaussian prior on the ratings.** Without it an undefeated
      entrant runs to infinity.

    Newton-Raphson on a concave objective, so it converges from zero. The
    `standard_errors` it returns come from the Hessian and are **not** what to
    publish: with a fractional score the objective is a quasi-likelihood, so
    they understate the true error. Use `bootstrap_ratings`.
    """
    names = tuple(entrants)
    if anchor not in names:
        raise ValueError(f"anchor {anchor!r} is not among the entrants {names}")
    if design.n_games == 0:
        raise ValueError("cannot fit ratings from an empty design")

    n_entrants = len(names)
    anchor_index = names.index(anchor)
    # Parameters are [ratings excluding the anchor, h_zone, h_turn, h_seat];
    # striking the anchor's column out is what pins it at zero.
    free = [i for i in range(n_entrants) if i != anchor_index]
    n_free = len(free)
    matrix = _design_matrix(design, n_entrants, free)
    _require_identifiable(matrix, n_free, prior_sigma)

    parameters = np.zeros(n_free + 3, dtype=np.float64)
    curvature = np.zeros((n_free + 3, n_free + 3), dtype=np.float64)
    converged = False
    iteration = 0
    # d(logit)/d(difference): the model is base-10 on a 400-point scale, so one
    # rating point is `ln(10)/400` of a natural logit.
    rate = _LOG_TEN / ELO_SCALE

    for iteration in range(1, max_iter + 1):
        difference = matrix @ parameters
        expected = _sigmoid(rate * difference)
        residual = design.score - expected

        gradient = rate * (matrix.T @ residual)
        weights = rate**2 * expected * (1.0 - expected)
        curvature = (matrix.T * weights) @ matrix

        # The prior applies to the ratings only -- the three advantage terms are
        # structural quantities of the scenario, not entrants to be shrunk.
        # `h_seat` in particular must NOT be shrunk: it is the term that absorbs
        # the seating confound, and shrinking it would push that confound back
        # into the ratings, silently.
        if prior_sigma > 0.0:
            precision = 1.0 / prior_sigma**2
            gradient[:n_free] -= parameters[:n_free] * precision
            curvature[:n_free, :n_free] += precision * np.eye(n_free)

        step = np.linalg.solve(curvature, gradient)
        parameters = parameters + step
        if float(np.max(np.abs(step))) < tol:
            converged = True
            break

    ratings = np.zeros(n_entrants, dtype=np.float64)
    ratings[free] = parameters[:n_free]

    standard_errors = np.zeros(n_entrants, dtype=np.float64)
    standard_errors[free] = np.sqrt(np.diag(np.linalg.inv(curvature))[:n_free])

    return RatingFit(
        entrants=names,
        ratings=ratings,
        h_zone=float(parameters[n_free]),
        h_turn=float(parameters[n_free + 1]),
        h_seat=float(parameters[n_free + 2]),
        standard_errors=standard_errors,
        n_games=design.n_games,
        n_iterations=iteration,
        converged=converged,
    )


def bootstrap_ratings(
    design: Design,
    entrants: Sequence[str],
    n_bootstrap: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
    **fit_kwargs: object,
) -> RatingTable:
    """Resample **layouts** with replacement and re-fit; percentile interval.

    Layouts rather than games, because the four legs played on one layout share
    their terrain, objectives and dice, and games sharing an entrant share its
    strength. They are one unit of evidence, not four, and resampling rows
    would treat them as independent.

    **That correlation is why the Hessian cannot be used, and `docs/elo.md`
    gives the wrong reason.** It argues that with a fractional score the
    Bernoulli objective is a quasi-likelihood whose implied variance
    *understates* the error. The first half is right and the conclusion is
    backwards: any `[0, 1]`-valued score has variance at most `p(1-p)`, so the
    Bernoulli assumption is an upper bound on the *marginal* variance and is
    conservative on its own -- which is the same fact as the
    Rao-Blackwellisation argument that motivates the score. Measured on
    synthetic data with independent rows, the Hessian interval comes out
    **wider** than the bootstrap, not narrower.

    What the Hessian genuinely misses is **dependence between rows**, which no
    per-row variance assumption can see, and which is unbounded in the
    direction that matters. Hence a layout bootstrap, and hence
    `test_the_layout_bootstrap_is_wider_than_resampling_rows`.

    Seeded, so a published table reproduces.
    """
    point = fit_ratings(design, entrants, **fit_kwargs)  # type: ignore[arg-type]
    rows_by_layout = _rows_by_layout(design)
    layouts = np.array(sorted(rows_by_layout), dtype=np.intp)
    rng = np.random.default_rng(seed)

    ratings = np.empty((n_bootstrap, len(point.entrants)), dtype=np.float64)
    advantages = np.empty((n_bootstrap, 3), dtype=np.float64)
    for draw in range(n_bootstrap):
        drawn = rng.choice(layouts, size=layouts.size, replace=True)
        rows = np.concatenate([rows_by_layout[int(layout)] for layout in drawn])
        resampled = fit_ratings(_take(design, rows), entrants, **fit_kwargs)  # type: ignore[arg-type]
        ratings[draw] = resampled.ratings
        advantages[draw] = (resampled.h_zone, resampled.h_turn, resampled.h_seat)

    low, high = 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)
    zone_interval = np.percentile(advantages[:, 0], [low, high])
    turn_interval = np.percentile(advantages[:, 1], [low, high])
    seat_interval = np.percentile(advantages[:, 2], [low, high])

    return RatingTable(
        fit=point,
        lower=np.percentile(ratings, low, axis=0),
        upper=np.percentile(ratings, high, axis=0),
        h_zone_interval=(float(zone_interval[0]), float(zone_interval[1])),
        h_turn_interval=(float(turn_interval[0]), float(turn_interval[1])),
        h_seat_interval=(float(seat_interval[0]), float(seat_interval[1])),
        n_bootstrap=n_bootstrap,
        alpha=alpha,
    )


def _design_matrix(
    design: Design, n_entrants: int, free: Sequence[int]
) -> NDArray[np.float64]:
    """Rows of `[+1 A, -1 B, sigma_zone, sigma_turn, 1]`, anchor dropped.

    Dense on purpose: a 100-layout, 9-entrant table is ~14k rows by ~11 columns,
    which is well under a megabyte and lets the Hessian be one `matmul`.

    Built with `np.add.at` rather than assignment so that a **self-pairing**
    accumulates to zero instead of leaving a stray `+1`. That case is not
    hypothetical: an entrant played against itself is the seat-parity check, and
    it is the cleanest estimator of all three advantages there is -- the rating
    difference is identically zero by construction, so whatever margin survives
    the balanced four legs is structural and nothing else.

    The final column is **constant at 1**, not read off the design, because the
    arena always seats entrant A on the player seat. It is a column rather than
    an intercept-by-another-name for a reason: it means the seat advantage is a
    *parameter with an interval* instead of a bias smeared across the ratings in
    proportion to how often each entrant happened to be named first. Should a
    schedule ever swap the seats, this becomes data and moves onto `Design`.
    """
    full = np.zeros((design.n_games, n_entrants + 3), dtype=np.float64)
    rows = np.arange(design.n_games)
    np.add.at(full, (rows, design.index_a), 1.0)
    np.add.at(full, (rows, design.index_b), -1.0)
    full[:, n_entrants] = design.sigma_zone
    full[:, n_entrants + 1] = design.sigma_turn
    full[:, n_entrants + 2] = 1.0
    keep = list(free) + [n_entrants, n_entrants + 1, n_entrants + 2]
    return full[:, keep]


def _require_identifiable(
    matrix: NDArray[np.float64], n_free: int, prior_sigma: float
) -> None:
    """Refuse a design that cannot separate the parameters it is asked for.

    The prior makes the *rating* block full rank on its own, but nothing
    regularises the three advantage terms -- they are structural quantities of
    the scenario rather than entrants to be shrunk. So a schedule that varies
    the zone and turn axes together leaves that pair identified only up to their
    sum, and Newton hits a singular Hessian.

    Refusing is the point. Regularising instead would return a plausible split
    of a quantity the data never separated, and a rating that reports a
    deployment-zone advantage it could not have measured is worse than no
    rating. The four-leg schedule exists precisely so this cannot happen.

    **`h_seat` fails differently, and the prior makes it worse rather than
    better.** Its column is constant, so with two entrants and no self-pairing
    it is exactly collinear with the single free rating: the null direction has
    a rating component, which means the prior *does* make the Hessian
    invertible, and the split between "B is weaker" and "the player seat is
    stronger" would then be decided by the prior rather than by any game. That
    is the failure this function exists to prevent, so the rank check for the
    seat column runs **whether or not there is a prior**.
    """
    advantages = matrix[:, n_free:]
    if np.linalg.matrix_rank(advantages) < advantages.shape[1]:
        raise ValueError(
            "the schedule does not identify the zone and first-turn advantages "
            "separately -- they vary together, so only their sum is measurable. "
            "Play all four legs (zone x first mover), not two."
        )
    if np.linalg.matrix_rank(matrix) < matrix.shape[1]:
        without_seat = matrix[:, :-1]
        if np.linalg.matrix_rank(without_seat) == without_seat.shape[1]:
            raise ValueError(
                "this table cannot separate the player-seat advantage from the "
                "ratings themselves. Entrant A always takes the player seat, so "
                "the seat term is identified only through a cycle in the pairing "
                "graph -- which needs at least THREE entrants -- or through a "
                "self-pairing, one entrant on both seats, which is what "
                "`just measure-seat-parity` plays. Add a third entrant, or feed "
                "the seat-parity legs into this ledger."
            )
        if prior_sigma <= 0.0:
            raise ValueError(
                "the design does not identify every rating; with no prior, an "
                "entrant needs at least one game against the connected pool"
            )


def _rows_by_layout(design: Design) -> dict[int, NDArray[np.intp]]:
    """Row indices grouped by layout, so a resample can gather whole layouts."""
    order = np.argsort(design.layout, kind="stable")
    sorted_layouts = design.layout[order]
    boundaries = np.flatnonzero(np.diff(sorted_layouts)) + 1
    groups = np.split(order, boundaries)
    keys = np.split(sorted_layouts, boundaries)
    return {int(key[0]): rows for key, rows in zip(keys, groups) if key.size}


def _take(design: Design, rows: NDArray[np.intp]) -> Design:
    """A sub-design over the given rows, preserving layout identity."""
    return Design(
        index_a=design.index_a[rows],
        index_b=design.index_b[rows],
        sigma_zone=design.sigma_zone[rows],
        sigma_turn=design.sigma_turn[rows],
        score=design.score[rows],
        layout=design.layout[rows],
    )


def _sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """`1 / (1 + exp(-x))`, stable for large negative `x`."""
    positive = values >= 0.0
    result = np.empty_like(values)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result
