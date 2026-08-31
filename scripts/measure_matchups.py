"""Which of our units wants to meet which of theirs, before a model has moved.

The pre-game read: for every pair of units, what one round of fire does, how
long the target survives it, and how much of the game the longer-ranged unit
gets for free. Static -- it reads the config and nothing else, so it costs no
episodes, no dice and no GPU, and it answers before deployment rather than after.

The per-model expected-damage matrix already ships and is already an observation
input (`domain/shooting.py::expected_damage_matrix`, hstacked onto every player
model token). This is the **unit-level reduction of that same matrix**, not a
second calculation: all shooters in a unit fire so the attacker axis sums, one
representative defender takes the maths so the defender axis does not. Computed
any other way, the table a human reads and the number the network sees could
disagree.

HOW TO READ IT

* `cas/rd` is expected models removed per round of that unit's fire, in the
  **open** and capped at the target unit's size. Cover is not applied, matching
  the shipped matrix.
* `overkill` is the share of expected wounds the cap discards. A bound cap is
  fire being wasted, and the answer to it is to split the volley -- the value of
  which is a step function on control counts, not a smooth efficiency gain.
* `free` is rounds of unanswered fire while the shorter-ranged unit closes:
  `(reach advantage) / (both Moves)`. Kinematic and position-free.
* `to halve` is rounds to remove half the target unit, ignoring return fire.

⚠ **RANGE IS NEVER FOLDED INTO `cas/rd`.** This tool has no positions. Reach
appears only as `reach` and `free`, and the exchange ratio is quoted at two
distances rather than blended into one, because a single number would hide which
distance chose it.

⚠ **ON THE CONFIG THAT TRAINS THIS TABLE IS 1x1.** Both armies in
`configs/golden/25v25_maps_two_mode.yaml` are one profile. It says something
only where the profiles differ -- `30v15_fast_horde_vs_elite`, the mixed-role
arms -- and reading a 1x1 table as a finding is how it gets oversold.

⚠ **Expected damage is DAMAGE POINTS; casualties are models removed.** Two
things separate them, and reaching for only the first is the mistake: excess
damage is lost (a Damage-3 hit on a Wounds-1 model removes one model and wastes
two points), and a model may take several hits (a Wounds-3 model absorbs three
Damage-1 hits). A model costs `D x ceil(W / D)` points either way round. Every
shipped config is `damage: 1` against `max_wounds: 1`, where the two columns are
the same number and neither effect is visible.

Usage: just measure-matchups <env_config> [key=value...]
"""

from __future__ import annotations

import sys

from scripts.scenario_overrides import describe, load_env_config, parse_overrides
from wargame_rl.wargame.envs.board.matchup import (
    UnitProfile,
    exchange_ratio,
    matchup_table,
    unit_profiles,
)
from wargame_rl.wargame.envs.types.config import WargameEnvConfig


def _profiles(
    config: WargameEnvConfig,
) -> tuple[tuple[UnitProfile, ...], tuple[UnitProfile, ...]]:
    """Both sides' units, or empty tuples for a side the config does not arm."""
    player = unit_profiles(
        config.models, config.number_of_wargame_models, config.max_move_speed
    )
    opponent = unit_profiles(
        config.opponent_models,
        config.number_of_opponent_models,
        config.max_move_speed,
    )
    return player, opponent


def _by_profile(
    profiles: tuple[UnitProfile, ...],
) -> tuple[tuple[UnitProfile, tuple[int, ...]], ...]:
    """Group units by their stat line, keeping one representative of each.

    Almost every army here is several units of one profile, so a row per unit
    pair prints the same numbers thirty times and buries the one line that
    differs. Collapsing to distinct profiles makes an asymmetric config
    readable and makes a symmetric one obviously symmetric.
    """
    seen: dict[tuple, list[UnitProfile]] = {}
    for unit in profiles:
        key = (
            unit.move,
            unit.weapon_range,
            unit.attacks,
            unit.ballistic_skill,
            unit.strength,
            unit.ap,
            unit.damage,
            unit.toughness,
            unit.save,
            unit.max_wounds,
            unit.n_models,
        )
        seen.setdefault(key, []).append(unit)
    return tuple(
        (members[0], tuple(u.group_id for u in members)) for members in seen.values()
    )


def _units_label(group_ids: tuple[int, ...]) -> str:
    """`x3 (u0-u2)` -- how many units share this profile, and which."""
    if len(group_ids) == 1:
        return f"u{group_ids[0]}"
    contiguous = list(group_ids) == list(range(group_ids[0], group_ids[-1] + 1))
    span = (
        f"u{group_ids[0]}-u{group_ids[-1]}"
        if contiguous
        else ",".join(f"u{g}" for g in group_ids)
    )
    return f"x{len(group_ids)} ({span})"


def _describe_units(label: str, profiles: tuple[UnitProfile, ...]) -> None:
    if not profiles:
        print(f"\n{label}: no units (the config declares no models for this side)")
        return
    print(
        f"\n{label} -- {len(profiles)} units, {sum(u.n_models for u in profiles)} models"
    )
    print(
        f"  {'units':>14} {'models':>7} {'move':>6} {'range':>6} "
        f"{'A':>3} {'BS':>3} {'S':>3} {'AP':>3} {'D':>3} {'T':>3} {'Sv':>3} {'W':>3}"
    )
    for unit, group_ids in _by_profile(profiles):
        print(
            f"  {_units_label(group_ids):>14} {unit.n_models:>7} {unit.move:>6.1f} "
            f"{unit.weapon_range:>6.1f} {unit.attacks:>3} {unit.ballistic_skill:>3} "
            f"{unit.strength:>3} {unit.ap:>3} {unit.damage:>3} "
            f"{unit.toughness:>3} {unit.save:>3} {unit.max_wounds:>3}"
        )


def _report_direction(
    label: str,
    attackers: tuple[UnitProfile, ...],
    defenders: tuple[UnitProfile, ...],
) -> None:
    """One block per firing side, one row per attacker-defender pair."""
    if not attackers or not defenders:
        return
    firing = _by_profile(attackers)
    taking = _by_profile(defenders)
    print(f"\n{label}")
    print(
        f"  {'attacker':>14} {'defender':>14} {'cas/rd':>8} {'wnd/rd':>8} "
        f"{'overkill':>9} {'to halve':>9} {'reach':>7} {'free':>6}"
    )
    rows = matchup_table(
        tuple(unit for unit, _ in firing), tuple(unit for unit, _ in taking)
    )
    for (_, attacker_ids), row in zip(firing, rows, strict=True):
        for (_, defender_ids), matchup in zip(taking, row, strict=True):
            halve = matchup.rounds_to_halve
            print(
                f"  {_units_label(attacker_ids):>14} {_units_label(defender_ids):>14} "
                f"{matchup.casualties_per_round:>8.3f} "
                f"{matchup.wounds_per_round:>8.3f} "
                f"{matchup.overkill_share:>8.1%} "
                f"{('inf' if halve == float('inf') else f'{halve:.1f}'):>9} "
                f"{matchup.reach_margin:>+7.1f} {matchup.free_rounds:>6.2f}"
            )


def _report_exchanges(
    player: tuple[UnitProfile, ...], opponent: tuple[UnitProfile, ...]
) -> None:
    """The trade both ways, quoted at the two distances that differ."""
    if not player or not opponent:
        return
    print("\nexchange ratio -- our casualties inflicted per theirs, per round")
    print(f"  {'ours':>14} {'theirs':>14} {'long range':>12} {'short range':>13}")
    for mine, my_ids in _by_profile(player):
        for theirs, their_ids in _by_profile(opponent):
            long_range, short_range = exchange_ratio(mine, theirs)
            print(
                f"  {_units_label(my_ids):>14} {_units_label(their_ids):>14} "
                f"{_ratio(long_range):>12} {_ratio(short_range):>13}"
            )


def _ratio(value: float) -> str:
    """`inf` is a real state of this game -- only one side can answer at all."""
    if value == float("inf"):
        return "only ours"
    if value == 0.0:
        return "only theirs"
    return f"{value:.2f}"


def main() -> None:
    """Print the unit-versus-unit matchup tables for one scenario."""
    argv, overrides = parse_overrides(sys.argv)
    if len(argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    config_path = argv[1]
    config = load_env_config(config_path, **overrides)
    player, opponent = _profiles(config)

    print(f"\n{config_path}{describe(overrides)}")
    print("open ground, no cover -- see the module docstring")
    _describe_units("ours", player)
    _describe_units("theirs", opponent)
    _report_direction("ours firing into theirs", player, opponent)
    _report_direction("theirs firing into ours", opponent, player)
    _report_exchanges(player, opponent)
    _warn_if_uninformative(player, opponent)


def _warn_if_uninformative(
    player: tuple[UnitProfile, ...], opponent: tuple[UnitProfile, ...]
) -> None:
    """Say so when the scenario poses no matchup question at all.

    The uninformative case is a **mirror** -- one profile a side and the same
    profile on both -- where every cell of every table above is the same number
    and no allocation follows from it. Two sides that differ are informative
    even at one profile each, which is exactly what the asymmetric configs are.
    """
    mine, theirs = _by_profile(player), _by_profile(opponent)
    if not mine or not theirs or len(mine) > 1 or len(theirs) > 1:
        return
    ours, others = mine[0][0], theirs[0][0]
    mirrored = ours.attacker_row == others.attacker_row and (
        ours.toughness,
        ours.save,
        ours.max_wounds,
        ours.move,
        ours.weapon_range,
    ) == (
        others.toughness,
        others.save,
        others.max_wounds,
        others.move,
        others.weapon_range,
    )
    if mirrored:
        print(
            "\n-> MIRROR: one profile a side and the same profile on both. Every "
            "cell above is the same number and no allocation follows from it. "
            "This table says something only where the profiles differ."
        )
    else:
        print(
            "\n-> asymmetric: the two sides differ, so the reach and exchange "
            "columns are the ones carrying information here."
        )


if __name__ == "__main__":
    main()
