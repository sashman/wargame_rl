"""The four-leg match schedule, expressed as config transforms.

A rated pairing plays every combination of the two axes the board is imbalanced
on -- who deploys in which zone, and who moves first:

| Leg | A's zone | Moves first |
|-----|----------|-------------|
| 1   | zone 1   | A           |
| 2   | zone 1   | B           |
| 3   | zone 2   | A           |
| 4   | zone 2   | B           |

Balanced in both axes by construction, so the ratings are unbiased even before
the advantage terms are fitted -- and *because* both axes vary independently,
both terms are identified. Varying them together would confound them into one
number from which neither is recoverable, which `fit_ratings` refuses outright.

**Neither axis needs new environment code.** `turn_order` is already a config
field read only by `_resolve_player_side`, and A's zone is a swap of the two
deployment-zone fields. Entrant A always sits on the player seat; entrant B
rides in `opponent_policy`.

⚠ **The engine SEAT is a third axis, and these four legs do not balance it** --
see `pairings` below and `docs/elo.md` § Open gaps. "Unbiased before the
advantage terms are fitted" holds for zone and turn order, and for the seat only
on a scenario that passes `just measure-seat-parity`.

This module may import `envs/types` -- the shared kernel -- and nothing else
from the project.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import cast

from wargame_rl.wargame.envs.types import (
    OpponentPolicyConfig,
    TurnOrder,
    WargameEnvConfig,
)

# Disjoint from rollout (0), in-run baselines (10_000), in-run eval (500_000),
# held-out scoring (700_000) and behaviour cloning (800_000), so a rated match
# is played on layouts nothing else in the repo reports on.
ELO_SEED_BASE = 900_000

# Offset rather than reuse: the dice are a bigger source of spread than the
# scenario (sd 50.6 within a layout against 45.0 between), so the combat stream
# is pinned separately and held fixed across a pairing's four legs.
COMBAT_SEED_OFFSET = 1_000_000


class Zone(str, Enum):
    """Which deployment zone entrant A occupies."""

    zone_1 = "zone_1"
    zone_2 = "zone_2"


class Seat(str, Enum):
    """Which entrant a leg attributes something to."""

    a = "a"
    b = "b"


@dataclass(frozen=True, slots=True)
class Leg:
    """One of the four balanced legs a rated pairing plays."""

    a_zone: Zone
    first_mover: Seat

    @property
    def sigma_zone(self) -> float:
        """`+1` when A deployed in zone 1, `-1` in zone 2."""
        return 1.0 if self.a_zone is Zone.zone_1 else -1.0

    @property
    def sigma_turn(self) -> float:
        """`+1` when A moved first, `-1` otherwise."""
        return 1.0 if self.first_mover is Seat.a else -1.0


FOUR_LEGS: tuple[Leg, Leg, Leg, Leg] = (
    Leg(Zone.zone_1, Seat.a),
    Leg(Zone.zone_1, Seat.b),
    Leg(Zone.zone_2, Seat.a),
    Leg(Zone.zone_2, Seat.b),
)

# The turn-order pair, for a scenario whose zone axis does nothing. Both sit in
# zone 1 so the recorded `sigma_zone` is honest about not having varied -- which
# is also why `fit_ratings` refuses such a design: a constant column cannot
# separate `h_zone` from `h_turn`. A caller playing these reports the aggregate
# and skips the Elo decomposition rather than fitting a term it did not vary.
ZONE_BLIND_LEGS: tuple[Leg, Leg] = (
    Leg(Zone.zone_1, Seat.a),
    Leg(Zone.zone_1, Seat.b),
)


class InertLegError(ValueError):
    """The config would make a leg's axis do nothing, silently."""


def config_for_leg(
    base: WargameEnvConfig, leg: Leg, require_live_zone_axis: bool = True
) -> WargameEnvConfig:
    """A copy of `base` set up for one leg.

    `require_live_zone_axis` is what a caller sets to **False** when it does not
    consume `h_zone`. `just measure-seat-parity` is the only such caller: its
    verdict is the aggregate margin over the legs, and the zone term is a line
    it prints beside that and never reads. A caller that does read `h_zone`
    leaves this True and is refused, which is the default.

    ⚠ **Turning it off does not make the axis work -- it makes the schedule
    honest about the axis being dead.** Pair it with `legs_for`, which drops to
    the turn-order pair: with an inert zone axis legs 1 and 3 are the same
    config on the same layout and combat seeds, so they are the *same episode*
    played twice, and averaging all four would count every game twice and
    understate the standard error by a factor of sqrt(2).

    Raises `InertLegError` rather than producing a config on which the zone axis
    does nothing. Three ways that happens, all silent:

    - **Zones left at their defaults.** `battle_factory.from_config` derives
      `(0, 0, W//3, H)` and `(W*2//3, 0, W, H)` when the fields are `None`, so
      swapping two `None`s changes nothing. The fit would then attribute
      whatever noise it found to `h_zone` and report it as a measurement.
    - **Fixed model positions.** With explicit per-model coordinates the zones
      are decorative and the armies deploy where they are told regardless.
    - **A map pool.** A drawn table carries its own deployment *outlines*
      (`TerrainMapConfig.deployment`), and those govern placement's sampling
      *and* its acceptance; the config rectangles are ignored outright. So the
      swap is a **total no-op** -- measured on `25v25_maps_two_mode.yaml`, both
      armies' positions and the outline itself are bit-identical across it on 10
      of 10 layouts. This is the same failure as the `None` case above, reached
      by a different route.

    The map-pool case is why this check exists in its current form: it was
    written when the rectangles were the zones, and the tables were regenerated
    with their own polygons three days later. Rating a pool config in between
    would have fitted `h_zone` from noise on the config that trains.

    ⚠ **`h_zone`'s true value is ZERO by construction on this pool**, which is a
    stronger statement than "unmeasured": the two deployment outlines are 180
    degree rotations of each other on **45 of 45** tables. There is nothing for
    the axis to measure even in principle. (The *terrain* is rotation-invariant
    on only 34 of 45, so the two SEATS still face different games on eleven
    tables -- that asymmetry is `h_seat`'s to carry, not `h_zone`'s, and on a
    pool config the seat is perfectly confounded with the side of the table.)

    ⚠ The refusal is on `map_pool` itself, not on whether its maps carry
    outlines -- this module may import `envs/types` and nothing else, so it
    cannot read the map files. A pool whose maps all leave `deployment` unset
    would be safe to rate and is refused anyway. No shipped pool is like that
    (54 of 54 map files carry one), and over-refusing costs a config while
    under-refusing costs a published number.

    Refusing is the point: a rating that reports a deployment-zone advantage it
    never varied is worse than no rating at all.
    """
    if not require_live_zone_axis:
        return _with_turn_order(base, leg)
    if base.deployment_zone is None or base.opponent_deployment_zone is None:
        raise InertLegError(
            "both deployment_zone and opponent_deployment_zone must be set "
            "explicitly to rate on this config: the defaults are derived at "
            "battle construction, so swapping two Nones is a silent no-op and "
            "the zone advantage would be fitted from noise"
        )
    if base.has_fixed_model_positions or base.has_fixed_opponent_positions:
        raise InertLegError(
            "fixed model positions make the zone swap inert -- the armies "
            "deploy where they are told regardless of which zone is whose"
        )
    if base.map_pool is not None:
        raise InertLegError(
            "a map pool draws its own deployment outlines per episode and "
            "ignores the config rectangles entirely, so swapping them is a "
            "total no-op and h_zone would be fitted from noise. On this pool "
            "its true value is zero anyway -- the two outlines are 180-degree "
            "rotations of each other on 45 of 45 tables. Rate a config that "
            "deploys under its own deployment_zone rectangles, or teach the "
            "pool to swap its outlines first"
        )

    config = _with_turn_order(base, leg)
    if leg.a_zone is Zone.zone_2:
        config.deployment_zone = base.opponent_deployment_zone
        config.opponent_deployment_zone = base.deployment_zone
    return config


def _with_turn_order(base: WargameEnvConfig, leg: Leg) -> WargameEnvConfig:
    """A render-free copy of `base` carrying only the leg's turn order.

    The whole of a leg on a config with no live zone axis, and the first half of
    one on a config that has it.
    """
    config = cast(WargameEnvConfig, base.model_copy(deep=True))
    config.render_mode = None
    config.turn_order = (
        TurnOrder.player if leg.first_mover is Seat.a else TurnOrder.opponent
    )
    return config


def legs_for(require_live_zone_axis: bool = True) -> tuple[Leg, ...]:
    """The legs a pairing plays, given whether its zone axis does anything.

    Four when it does. **Two when it does not** -- the turn-order pair, both in
    zone 1. Playing all four on a dead axis does not merely waste half the
    games: legs 1 and 3 become the same config on the same layout and combat
    seeds, so they are bit-identical episodes, and an aggregate over all four
    counts every game twice while its standard error assumes it did not.
    """
    return FOUR_LEGS if require_live_zone_axis else ZONE_BLIND_LEGS


def with_opponent(
    config: WargameEnvConfig, opponent: OpponentPolicyConfig
) -> WargameEnvConfig:
    """A copy of `config` with entrant B seated as the opponent policy.

    Kept apart from `config_for_leg` because they are different concerns: a leg
    is the two imbalance axes, while the opponent is *who is being rated*. The
    ledger's scenario fingerprint drops `opponent_policy` and `turn_order` for
    exactly that reason.
    """
    updated = cast(WargameEnvConfig, config.model_copy(deep=True))
    updated.opponent_policy = opponent
    return updated


def layout_seeds(n_layouts: int, seed_base: int = ELO_SEED_BASE) -> list[int]:
    """The layout seeds a rated pairing plays, in order.

    The same seeds for every entrant and every leg, so a difference between two
    rows is the policy and not the draw.
    """
    if n_layouts <= 0:
        raise ValueError(f"n_layouts must be positive, got {n_layouts}")
    return [seed_base + index for index in range(n_layouts)]


def combat_seeds_for(seeds: Sequence[int]) -> list[int]:
    """Dice seeds, held fixed across a pairing's four legs.

    The four legs of one layout then share their initial conditions *and* their
    opening rolls, so the comparison is as paired as it can be made. The stream
    diverges once the actions do; that is unavoidable and is what the layout
    bootstrap covers.
    """
    return [seed + COMBAT_SEED_OFFSET for seed in seeds]


def pairings(entrants: Sequence[str]) -> list[tuple[str, str]]:
    """Every unordered pair, each listed once, in input order.

    Ordered pairs would double the cost: the four legs already balance both
    axes, so playing B-versus-A adds no information A-versus-B lacks.

    ⚠ **That holds only while the two SEATS are the same game**, which is a
    third axis the four legs do not balance -- entrant A always sits on the
    player seat, so the entrant named first on the command line takes it in
    every one of its pairings and the entrant named last never takes it at all.
    Measured on `configs/golden/25v25_shooting_opponent.yaml` the seats differ
    by **-24.6 +/- 9.4 vp**, which would make ratings there a function of
    argument order. `just measure-seat-parity` is the gate, and nothing calls
    it. See `docs/elo.md` § Open gaps.
    """
    names = list(entrants)
    if len(set(names)) != len(names):
        raise ValueError(f"entrants must be distinct, got {names}")
    return [
        (names[first], names[second])
        for first in range(len(names))
        for second in range(first + 1, len(names))
    ]
