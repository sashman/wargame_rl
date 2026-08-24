"""Turn played legs into a fitted, printable rating table."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from wargame_rl.wargame.rating.arena import LegResult
from wargame_rl.wargame.rating.elo import Design, RatingTable, bootstrap_ratings
from wargame_rl.wargame.rating.score import DEFAULT_MARGIN_SCALE, margin_score

DEFAULT_ANCHOR = "random"


def design_from_legs(
    legs: Sequence[LegResult], margin_scale: float = DEFAULT_MARGIN_SCALE
) -> tuple[Design, tuple[str, ...]]:
    """Flatten legs into one row per game, with layouts kept as the bootstrap unit.

    Layout identity is the **seed**, not the row index, so two legs played on
    the same seed resample together however they were ordered on disk.
    """
    if not legs:
        raise ValueError("cannot build a design from no legs")

    entrants = tuple(
        sorted({name for leg in legs for name in (leg.entrant_a, leg.entrant_b)})
    )
    position = {name: index for index, name in enumerate(entrants)}
    layout_id: dict[int, int] = {}

    index_a: list[int] = []
    index_b: list[int] = []
    sigma_zone: list[float] = []
    sigma_turn: list[float] = []
    margins: list[float] = []
    layout: list[int] = []

    for leg in legs:
        for seed, margin in zip(leg.layout_seeds, leg.margins):
            index_a.append(position[leg.entrant_a])
            index_b.append(position[leg.entrant_b])
            sigma_zone.append(leg.leg.sigma_zone)
            sigma_turn.append(leg.leg.sigma_turn)
            margins.append(margin)
            layout.append(layout_id.setdefault(int(seed), len(layout_id)))

    design = Design(
        index_a=np.array(index_a, dtype=np.intp),
        index_b=np.array(index_b, dtype=np.intp),
        sigma_zone=np.array(sigma_zone, dtype=np.float64),
        sigma_turn=np.array(sigma_turn, dtype=np.float64),
        score=margin_score(np.array(margins, dtype=np.float64), margin_scale),
        layout=np.array(layout, dtype=np.intp),
    )
    return design, entrants


def rate(
    legs: Sequence[LegResult],
    margin_scale: float = DEFAULT_MARGIN_SCALE,
    anchor: str = DEFAULT_ANCHOR,
    n_bootstrap: int = 500,
    seed: int = 0,
) -> RatingTable:
    """Fit a rating table from played legs, with bootstrap intervals."""
    design, entrants = design_from_legs(legs, margin_scale)
    if anchor not in entrants:
        raise ValueError(
            f"anchor {anchor!r} did not play; a table needs a fixed reference in "
            f"it or 'zero Elo' means nothing. Entrants were {list(entrants)}"
        )
    return bootstrap_ratings(
        design, entrants, n_bootstrap=n_bootstrap, seed=seed, anchor=anchor
    )


def mean_margins(legs: Sequence[LegResult]) -> dict[str, float]:
    """Mean `vp_margin` per entrant, over every game it played.

    Printed beside the ratings because **Elo ranks, it does not explain**. It is
    also the cross-check: near equality `dR ~ 173.7 * mean_margin / s_m`, so a
    rating that disagrees badly with this column is measuring something else.
    """
    totals: dict[str, list[float]] = {}
    for leg in legs:
        totals.setdefault(leg.entrant_a, []).extend(leg.margins)
        totals.setdefault(leg.entrant_b, []).extend(-margin for margin in leg.margins)
    return {name: float(np.mean(values)) for name, values in totals.items()}


def mean_coherency(legs: Sequence[LegResult]) -> dict[str, float | None]:
    """Mean intended coherency per entrant, for the games it played as A.

    Unconditional, because a `vp_margin` on its own is a result plus an unstated
    claim that the moves earning it were legal, and only this column carries the
    claim.

    ⚠ **It is `None` for an entrant only ever seated as B**, because
    `evaluate_selector` measures the player seat and nothing measures the
    opponent's. `pairings` lists each pair once in input order, so the entrant
    named *last* on the command line is entrant B in every one of its pairings
    and comes back with no coherency at all -- a real gap against this repo's
    rule that no score is quoted without it.

    `CoherencyTracker` is already written as running totals for *one force*, so
    a second instance is mechanically small; what is missing is the opponent's
    **intended** coherency, which the player path reads off
    `ActionHandler.intended_coherency_last_move` and the opponent path does not
    record. See `docs/elo.md` § Open gaps -- this is the same seating asymmetry
    that leaves the engine seat unbalanced, seen from the metrics side.
    """
    totals: dict[str, list[float]] = {}
    for leg in legs:
        if leg.coherency_rate is not None:
            totals.setdefault(leg.entrant_a, []).append(leg.coherency_rate)
    return {name: float(np.mean(values)) for name, values in totals.items()}


def format_table(
    table: RatingTable,
    legs: Sequence[LegResult],
    margin_scale: float = DEFAULT_MARGIN_SCALE,
) -> str:
    """The printable table: rating, interval, margin and coherency per entrant."""
    margins = mean_margins(legs)
    coherency = mean_coherency(legs)
    order = np.argsort(-table.fit.ratings)

    header = (
        f"{'entrant':<28}{'Elo':>9}{'95% interval':>20}"
        f"{'vp margin':>12}{'coherent':>11}{'games':>8}"
    )
    lines = [header, "-" * len(header)]
    played = _games_per_entrant(legs)
    for index in order:
        name = table.fit.entrants[index]
        interval = f"[{table.lower[index]:+.0f}, {table.upper[index]:+.0f}]"
        coherent = coherency.get(name)
        lines.append(
            f"{name:<28}{table.fit.ratings[index]:>+9.0f}{interval:>20}"
            f"{margins.get(name, float('nan')):>+12.1f}"
            f"{('-' if coherent is None else f'{coherent:.3f}'):>11}"
            f"{played.get(name, 0):>8}"
        )

    lines.append("-" * len(header))
    lines.append(
        f"zone advantage   {table.fit.h_zone:+.1f} Elo  "
        f"[{table.h_zone_interval[0]:+.1f}, {table.h_zone_interval[1]:+.1f}]"
    )
    lines.append(
        f"first-turn adv.  {table.fit.h_turn:+.1f} Elo  "
        f"[{table.h_turn_interval[0]:+.1f}, {table.h_turn_interval[1]:+.1f}]"
    )
    lines.append(
        f"\n{table.fit.n_games} games, margin scale {margin_scale:g}, "
        f"{table.n_bootstrap} bootstrap resamples over layouts"
    )
    lines.append(
        "quote the interval, not the point: a rating without one is the same "
        "failure as a success_rate with no floor and no bar"
    )
    return "\n".join(lines)


def _games_per_entrant(legs: Sequence[LegResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for leg in legs:
        for name in (leg.entrant_a, leg.entrant_b):
            counts[name] = counts.get(name, 0) + len(leg.margins)
    return counts
