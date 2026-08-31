"""Is the player seat advantaged, beyond the zone and the first turn?

**No rating on this config means anything until this reads zero.** The reward,
coherency and exposure trackers all sample the player's army only, and
`terminate_on_player_elimination` is a player-side option. Those are
measurement-only or default-off, so the *game* should be seat-symmetric -- but
that is a claim, not a fact, and if it is false then every Elo number on this
scenario is measuring an implementation asymmetry instead of a policy.

One policy plays **both seats** over the balanced four legs. Because it is the
same policy, its rating difference is identically zero by construction, so
whatever margin survives is the seat advantage and nothing else. That makes a
self-pairing the cleanest estimator of all three structural advantages there is
-- cleaner than any table over distinct entrants, where they are fitted
alongside every rating at once.

**The legs are appended to this scenario's ledger**, because they are the best
evidence of `h_seat` there is: entrant A always takes the player seat, so in a
table of distinct entrants that term is identified only through a cycle in the
pairing graph and comes back at 2-3x the standard error of `h_turn`. One
self-pairing roughly halves that. Throwing these legs away and then fitting the
seat term from cycles would be waste.

What to look for:

  aggregate margin ~ 0    the four legs cancel, so the seat itself is fair
  zone / first-turn       real structural advantages; a table CORRECTS for
                          these, so a non-zero value here is a finding, not a
                          failure
  player-seat adv.        the same quantity as the aggregate, in Elo rather
                          than victory points. A table corrects for it too --
                          but only once it has been measured, which is what
                          these legs are for

A non-zero **aggregate** margin is a bug, and it blocks rating this scenario.

Usage: just measure-seat-parity <env_config> [policy] [n_layouts]
"""

from __future__ import annotations

import sys

import numpy as np
from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.rating.arena import play_pairing, require_symmetric
from wargame_rl.wargame.rating.elo import fit_ratings
from wargame_rl.wargame.rating.entrant import Entrant
from wargame_rl.wargame.rating.ledger import append, fingerprint, path_for
from wargame_rl.wargame.rating.table import design_from_legs
from wargame_rl.wargame.selectors import build_action_selector


def main() -> None:
    """Play one policy against itself and report whether the seats are fair."""
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)

    config_path = sys.argv[1]
    policy = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else "squad_march_shoot"
    n_layouts = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else 30

    with open(config_path) as handle:
        config = parse_yaml_raw_as(WargameEnvConfig, handle.read())
    config.render_mode = None
    require_symmetric(config)

    entrant = Entrant(
        name=policy,
        build=lambda env: build_action_selector(policy, env).select,
        kind="baseline",
    )
    # A drawn table brings its own deployment outlines and ignores the config
    # rectangles, so the zone swap is a total no-op there. The gate does not
    # consume `h_zone` -- its verdict is the aggregate -- so it plays the
    # turn-order pair and declines to report a term it never varied.
    zone_axis_live = config.map_pool is None
    legs = play_pairing(
        entrant, entrant, config, n_layouts, require_live_zone_axis=zone_axis_live
    )

    print(f"\n{config_path}   {policy} on both seats")
    print(f"{n_layouts} layouts x {len(legs)} legs\n")

    header = f"{'leg':<28}{'mean margin':>13}{'wins':>8}"
    print(header)
    print("-" * len(header))
    for leg in legs:
        zone = "zone 1" if leg.leg.sigma_zone > 0 else "zone 2"
        first = "A first" if leg.leg.sigma_turn > 0 else "B first"
        print(
            f"{zone + ', ' + first:<28}{float(np.mean(leg.margins)):>+13.1f}"
            f"{float(np.mean(leg.wins)):>8.2f}"
        )
    print("-" * len(header))

    every_margin = np.array(
        [margin for leg in legs for margin in leg.margins], dtype=np.float64
    )
    aggregate = float(np.mean(every_margin))
    error = float(np.std(every_margin, ddof=1) / np.sqrt(every_margin.size))
    print(f"{'aggregate':<28}{aggregate:>+13.1f}   +/- {error:.1f} (1 se)")

    if zone_axis_live:
        design, entrants = design_from_legs(legs)
        fit = fit_ratings(design, entrants, anchor=policy)
        print(f"\nzone advantage   {fit.h_zone:+.1f} Elo")
        print(f"first-turn adv.  {fit.h_turn:+.1f} Elo")
        print(f"player-seat adv. {fit.h_seat:+.1f} Elo")
    else:
        print("\nzone advantage   -- not measurable here")
        print(
            "                 this config draws from a map pool, and a drawn "
            "table brings its\n                 own deployment outlines: the "
            "config rectangles are ignored, so\n                 swapping them "
            "changes nothing. The two outlines are 180-degree\n                 "
            "rotations of each other on 45 of 45 tables, so the advantage is\n"
            "                 zero by construction rather than merely unmeasured."
        )
        print("\nfirst-turn adv.  -- not reported\nplayer-seat adv. -- not reported")
        print(
            "                 the Elo decomposition needs all four legs; with a "
            "dead zone\n                 axis the design has a constant column "
            "and `fit_ratings` refuses\n                 it, which is correct. "
            "The aggregate above is the verdict."
        )

    verdict = (
        "the seats are fair; rating this scenario is sound"
        if abs(aggregate) <= 2.0 * error
        else "⚠ THE PLAYER SEAT IS ADVANTAGED -- every rating here would carry it"
    )
    print(f"\n{verdict}")
    print(
        "the zone and first-turn advantages are structural and a rating "
        "corrects for them; only the aggregate has to be zero"
    )

    if zone_axis_live:
        append(legs, config, [entrant])
        print(
            f"\nappended {len(legs)} legs to {path_for(fingerprint(config))} -- "
            "they are the direct evidence for the seat term in `just elo-table`"
        )
    else:
        print(
            "\nNOT appended to a ledger -- these legs never varied the zone, so "
            "a later fit\nover them would report an `h_zone` it could not have "
            "measured. The gate still\nstands on its own: the aggregate is what "
            "decides whether the seats are fair."
        )


if __name__ == "__main__":
    main()
