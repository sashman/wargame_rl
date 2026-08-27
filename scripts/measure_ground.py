"""Read the table before the game: who reaches which objective, and whose it was.

Two questions per table, both answered from the deployment alone, both static --
no episodes, no dice, no GPU.

**Whose ground is it.** Every objective is sorted into own-zone, contested and
hostile from the deployment outlines. The rules define no other board regions,
so this is derived from the zones and nothing else; 34 of the 45 real tables
have non-rectangular zones and `long_edges` splits the SHORT axis, so a
board-half rule would mean a different thing on every table.

**Who gets there first.** For each objective, the earliest round each side's
fastest unit could be standing on it, and the margin between them. This is the
quantity `measure-critic-probe` showed the critic cannot rank: it knows
redistribution is worth something and has no grip on which redistribution pays,
and arrival order is a distance and a divide.

It also puts a floor under a number `CLAUDE.md` flags as optimistic. The
`measure-objective-split` redistribution ceiling charges **no travel time and no
return fire**; the `+n rounds` column here is the travel half of that bill.

⚠ **ARRIVAL ROUNDS ARE A LOWER BOUND, three times over.** Coherency binds the
unit, so the straight line from its centroid is a bound and not a route.
Freezing means only ~92% of ordered inches are delivered. And nothing here
routes around a base or a terrain piece. Read a round as "not before this",
never as "then".

⚠ **A tie is not a win.** Control is `player_count > opponent_count`, strictly,
so a margin of 0 means arriving together and holding nothing.

Usage: just measure-ground <env_config> [maps_dir] [key=value...]
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.measure_maps import DEFAULT_MAPS_DIR, config_for_map, load_maps
from scripts.scenario_overrides import describe, load_env_config, parse_overrides
from wargame_rl.wargame.envs.board.reach import (
    ObjectiveReach,
    Ownership,
    objective_reach,
)
from wargame_rl.wargame.envs.board.threat import move_reach
from wargame_rl.wargame.envs.types.config import WargameEnvConfig
from wargame_rl.wargame.model.common.factory import create_environment

# One deployment per table. Placement is random within the zone, so this is a
# draw rather than the truth -- but the zones and the objectives are the table's,
# which is what the ownership column reads, and a unit centroid inside its own
# zone moves the arrival rounds by well under one.
LAYOUT_SEED = 700_000


def _read_table(config: WargameEnvConfig) -> tuple[ObjectiveReach, ...]:
    """Deploy once and price the ground."""
    env = create_environment(env_config=config)
    try:
        env.reset(seed=LAYOUT_SEED)
        return objective_reach(
            env,
            move_reach(config, config.models, len(env.player_models)),
            move_reach(config, config.opponent_models, len(env.opponent_models)),
        )
    finally:
        env.close()


def _rounds(value: float) -> str:
    return "-" if value == float("inf") else f"{value:.0f}"


def _margin(value: float) -> str:
    if value == float("inf"):
        return "ours"
    if value == -float("inf"):
        return "theirs"
    return f"{value:+.0f}"


def _print_table(name: str, reaches: tuple[ObjectiveReach, ...]) -> None:
    print(f"\n{name}")
    print(
        f"  {'obj':>4} {'x':>6} {'y':>6} {'zone':>10} "
        f"{'ours':>5} {'theirs':>7} {'margin':>7} {'first':>7}"
    )
    for reach in reaches:
        first = (
            "ours"
            if reach.contested_margin > 0
            else "theirs"
            if reach.contested_margin < 0
            else "tie"
        )
        print(
            f"  {reach.index:>4} {reach.location[0]:>6.1f} {reach.location[1]:>6.1f} "
            f"{reach.ownership.value:>10} {_rounds(reach.player_rounds):>5} "
            f"{_rounds(reach.opponent_rounds):>7} "
            f"{_margin(reach.contested_margin):>7} {first:>7}"
        )


def _summarise(all_reaches: list[tuple[ObjectiveReach, ...]], n_tables: int) -> None:
    """The distribution over tables, which is what a plan is actually made against."""
    flat = [reach for table in all_reaches for reach in table]
    if not flat:
        print("\nno objectives on any table")
        return
    zones = Counter(reach.ownership for reach in flat)
    margins = np.array(
        [
            reach.contested_margin
            for reach in flat
            if np.isfinite(reach.contested_margin)
        ]
    )
    ours = sum(1 for reach in flat if reach.contested_margin > 0)
    theirs = sum(1 for reach in flat if reach.contested_margin < 0)
    ties = len(flat) - ours - theirs

    print(f"\nover {n_tables} tables, {len(flat)} objectives")
    print(
        f"  {'own zone':>12} {zones[Ownership.own_zone]:>5} "
        f"({zones[Ownership.own_zone] / len(flat):>5.1%})"
    )
    print(
        f"  {'contested':>12} {zones[Ownership.contested]:>5} "
        f"({zones[Ownership.contested] / len(flat):>5.1%})"
    )
    print(
        f"  {'hostile':>12} {zones[Ownership.hostile]:>5} "
        f"({zones[Ownership.hostile] / len(flat):>5.1%})"
    )
    print(f"  {'we arrive first':>18} {ours:>5} ({ours / len(flat):>5.1%})")
    print(f"  {'they arrive first':>18} {theirs:>5} ({theirs / len(flat):>5.1%})")
    print(f"  {'tie (holds nothing)':>18} {ties:>5} ({ties / len(flat):>5.1%})")
    if margins.size:
        print(f"  {'median margin':>18} {np.median(margins):>+5.0f} rounds")
    banked = np.mean(
        [sum(1 for r in table if r.contested_margin > 0) for table in all_reaches]
    )
    print(f"  {'objectives we reach first, per table':>38} {banked:>5.2f}")
    print(
        "\n-> a margin of 0 is a TIE, and a tie controls nothing: control is a "
        "strict comparison. Arrival rounds are a LOWER bound -- see the header."
    )


def main() -> None:
    """Print the ground read for one scenario, per table and in aggregate."""
    argv, overrides = parse_overrides(sys.argv)
    if len(argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    config_path = argv[1]
    maps_dir = Path(argv[2]) if len(argv) > 2 and argv[2] else DEFAULT_MAPS_DIR

    base_config = load_env_config(config_path, **overrides)
    maps = load_maps(maps_dir)

    print(f"\n{config_path}{describe(overrides)}")
    print(
        f"{len(maps)} tables from {maps_dir}, one deployment each at seed {LAYOUT_SEED}"
    )

    all_reaches: list[tuple[ObjectiveReach, ...]] = []
    for terrain_map in maps:
        reaches = _read_table(config_for_map(base_config, terrain_map))
        all_reaches.append(reaches)
        _print_table(terrain_map.name, reaches)
    _summarise(all_reaches, len(maps))


if __name__ == "__main__":
    main()
