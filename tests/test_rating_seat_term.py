"""The player-seat advantage is a fitted term, not an assumption.

Entrant A always sits on the player seat and `pairings` lists each pair once in
**input order**, so before this term existed the first-named entrant took the
player seat in every one of its pairings and the last-named never took it at
all. On a scenario where the seats differ -- and on
`configs/golden/25v25_shooting_opponent.yaml` they differ by 24.6 vp -- that
made a rating a function of the order entrants were typed in.

These tests pin the three properties that closes it: the confound is absorbed,
a design that cannot separate it is refused rather than silently regularised,
and a self-pairing is worth more than another entrant.
"""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.rating.elo import Design, fit_ratings

from .test_rating_elo import ENTRANTS, _synthetic_design

PLANTED = {"random": 0.0, "squad_march": 250.0, "squad_march_shoot": 500.0}


def test_a_rating_no_longer_depends_on_the_order_entrants_are_named_in() -> None:
    """The defect this term exists to close.

    Permuting the entrant list permutes which of them sits on the player seat
    most often. With the seat advantage fitted, the ratings come back the same
    however the list is ordered.
    """
    design = _synthetic_design(
        PLANTED, zone_advantage=0.0, turn_advantage=0.0, seat_advantage=60.0
    )
    reversed_order = tuple(reversed(ENTRANTS))
    swapped = Design(
        index_a=np.array([reversed_order.index(ENTRANTS[i]) for i in design.index_a]),
        index_b=np.array([reversed_order.index(ENTRANTS[i]) for i in design.index_b]),
        sigma_zone=design.sigma_zone,
        sigma_turn=design.sigma_turn,
        score=design.score,
        layout=design.layout,
    )

    forward = fit_ratings(design, ENTRANTS, anchor="random")
    backward = fit_ratings(swapped, reversed_order, anchor="random")

    for name in ENTRANTS:
        assert backward.ratings[reversed_order.index(name)] == pytest.approx(
            forward.ratings[ENTRANTS.index(name)], abs=1e-6
        )
    assert backward.h_seat == pytest.approx(forward.h_seat, abs=1e-6)


def test_a_planted_seat_advantage_is_recovered() -> None:
    design = _synthetic_design(
        PLANTED,
        zone_advantage=0.0,
        turn_advantage=0.0,
        seat_advantage=60.0,
        self_pairings=(1,),
    )

    fit = fit_ratings(design, ENTRANTS, anchor="random", prior_sigma=0.0)

    assert fit.h_seat == pytest.approx(60.0, abs=1e-6)
    for name, value in PLANTED.items():
        assert fit.ratings[ENTRANTS.index(name)] == pytest.approx(value, abs=1e-6)


def test_two_entrants_without_a_self_pairing_are_refused() -> None:
    """Rank-deficient by construction, and the prior would hide it.

    With two entrants the single free rating column is exactly `-1` where the
    seat column is `+1`, so the two are collinear. The null direction has a
    rating component, which means the prior *does* make the Hessian invertible
    -- and the split between "B is weaker" and "the player seat is stronger"
    would then be decided by the prior rather than by any game played. That is
    the failure mode this refusal exists for, so it fires with a prior in place.
    """
    design = _synthetic_design(
        {"random": 0.0, "squad_march": 200.0},
        zone_advantage=0.0,
        turn_advantage=0.0,
        entrants=("random", "squad_march"),
        n_layouts=20,
    )

    with pytest.raises(ValueError, match="player-seat advantage"):
        fit_ratings(design, ("random", "squad_march"), anchor="random")


def test_two_entrants_with_a_self_pairing_are_fitted() -> None:
    """A self-pairing is what `just measure-seat-parity` plays.

    One entrant on both seats has a rating difference of identically zero, so
    whatever survives its balanced four legs is structural and nothing else.
    That is direct evidence, and it identifies the term where the pairing graph
    cannot.
    """
    design = _synthetic_design(
        {"random": 0.0, "squad_march": 200.0},
        zone_advantage=0.0,
        turn_advantage=0.0,
        seat_advantage=45.0,
        entrants=("random", "squad_march"),
        self_pairings=(0,),
        n_layouts=20,
    )

    fit = fit_ratings(
        design, ("random", "squad_march"), anchor="random", prior_sigma=0.0
    )

    assert fit.h_seat == pytest.approx(45.0, abs=1e-6)
    assert fit.ratings[1] == pytest.approx(200.0, abs=1e-6)


def test_a_self_pairing_sharpens_the_seat_term() -> None:
    """Worth more than another entrant, which is why the gate feeds the ledger.

    Without one, the term is identified only through a cycle in the pairing
    graph and carries a share of the prior's shrinkage. The numbers here are
    the reason `format_table` says which kind of identification it had.
    """
    without = fit_ratings(
        _synthetic_design(
            PLANTED, zone_advantage=0.0, turn_advantage=0.0, seat_advantage=0.0
        ),
        ENTRANTS,
        anchor="random",
    )
    with_self = fit_ratings(
        _synthetic_design(
            PLANTED,
            zone_advantage=0.0,
            turn_advantage=0.0,
            seat_advantage=0.0,
            self_pairings=(1,),
        ),
        ENTRANTS,
        anchor="random",
    )

    assert abs(with_self.h_seat) < abs(without.h_seat)
    top = ENTRANTS.index("squad_march_shoot")
    assert abs(with_self.ratings[top] - 500.0) < abs(without.ratings[top] - 500.0)


@pytest.mark.parametrize(
    ("n_entrants", "self_pairings", "identified"),
    [(2, (), False), (2, (0,), True), (3, (), True), (4, (), True)],
)
def test_identification_follows_the_pairing_graph(
    n_entrants: int, self_pairings: tuple[int, ...], identified: bool
) -> None:
    """Three entrants suffice, two do not, and a self-pairing rescues two.

    A constant column lies in the span of the `e_A - e_B` columns only if some
    rating vector has every pairwise difference equal to one, which any cycle
    forbids -- around a triangle the differences sum to zero and the ones do
    not. This is that argument, checked rather than asserted.
    """
    names = tuple(f"entrant_{index}" for index in range(n_entrants))
    design = _synthetic_design(
        {name: 100.0 * index for index, name in enumerate(names)},
        zone_advantage=0.0,
        turn_advantage=0.0,
        entrants=names,
        self_pairings=self_pairings,
        n_layouts=8,
    )

    if identified:
        assert fit_ratings(design, names, anchor=names[0]).converged
    else:
        with pytest.raises(ValueError, match="player-seat advantage"):
            fit_ratings(design, names, anchor=names[0])
