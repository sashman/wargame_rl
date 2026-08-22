"""Load an env config with a scenario scalar overridden, from the command line.

Two questions want the *same* scenario at several settings of one number — does
the game survive the tabletop's five battle rounds, and what does a longer weapon
range do — and until now neither could be asked. Every measurement script parses
argv positionally and hands the YAML straight to `parse_yaml_raw_as`, and
`train.py` exposes no env-scenario option at all, so a comparison meant copying a
13 KB golden config and editing one line.

`measure_maps.config_for_map` already refused that trade for terrain, and its
reasoning transfers verbatim: copying a scenario per variation means every future
reward change has to be applied N times, and **the first one that is missed makes
the comparison measure a different game, silently**. So the variation is a
`model_copy` of one config, exactly as it is there.

## Why `key=value` rather than another positional

`measure_maps` shipped a bug where an omitted `maps_dir` collapsed and shifted
`decode_topk` into its place, which made `just measure-maps <p> <cfg> <n>` fail
outright. Five scripts already carry four or five optional positionals each;
adding a sixth to every one of them would be the same trap five more times.

An override is therefore any argv token containing `=`. It can sit anywhere,
positional arguments keep their positions whatever is passed, and a script that
receives none behaves exactly as it did.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.types.config import TurnOrder, WargameEnvConfig

# Every override is a scenario *scalar*: one number, one meaning, no derived
# state to keep in step. Anything that would need a second field updated with it
# belongs in a config file, where it can be reviewed.
KNOWN_OVERRIDES = ("rounds", "weapon_range", "turn_order")


def parse_overrides(argv: list[str]) -> tuple[list[str], dict[str, str]]:
    """Split argv into positional arguments and `key=value` overrides.

    Returns the positionals in their original order, so a caller's existing
    indexing is unchanged, and the overrides as a plain string mapping — the
    values are typed by `apply_overrides`, which knows what each one is.

    An unknown key is an error rather than a silently ignored token: a
    misspelled override that read as "no override" would score the wrong
    scenario and look exactly like a real result.
    """
    positional: list[str] = []
    overrides: dict[str, str] = {}
    for token in argv:
        if "=" not in token:
            positional.append(token)
            continue
        key, _, value = token.partition("=")
        if key not in KNOWN_OVERRIDES:
            raise SystemExit(
                f"unknown override '{key}'. Known: {', '.join(KNOWN_OVERRIDES)}"
            )
        overrides[key] = value
    return positional, overrides


def apply_overrides(config: WargameEnvConfig, **overrides: str) -> WargameEnvConfig:
    """Return a deep copy of `config` with the named scenario scalars replaced.

    The copy is deep because `weapon_range` reaches into every model's weapon
    list, and mutating those in place would edit the caller's config — which,
    for a script that loads one config and scores several policies against it,
    would make the second policy play a different game from the first.
    """
    varied = cast(WargameEnvConfig, config.model_copy(deep=True))

    if "rounds" in overrides:
        varied.number_of_battle_rounds = int(overrides["rounds"])

    if "turn_order" in overrides:
        # Pinnable because `turn_order: random` is drawn from the *layout* RNG,
        # so `measure-noise-floor` books first-player advantage under "the
        # scenario" rather than "the dice". At four scoring events instead of
        # nineteen that draw is a far larger share of the outcome, so the two
        # horizons cannot be compared without holding it still.
        varied.turn_order = TurnOrder(overrides["turn_order"])

    if "weapon_range" in overrides:
        weapon_range = int(overrides["weapon_range"])
        # Both armies. Reaching only `models` would set up an asymmetric
        # firefight while the caller believes it changed one number.
        for model in list(varied.models or ()) + list(varied.opponent_models or ()):
            for weapon in model.weapons:
                weapon.range = weapon_range

    return varied


def load_env_config(path: str | Path, **overrides: str) -> WargameEnvConfig:
    """Parse the config at `path`, applying any scenario overrides.

    With no overrides this is exactly `parse_yaml_raw_as(WargameEnvConfig, ...)`,
    which is what keeps every existing invocation of every measurement script
    byte-identical.
    """
    config = cast(
        WargameEnvConfig, parse_yaml_raw_as(WargameEnvConfig, Path(path).read_text())
    )
    if not overrides:
        return config
    return apply_overrides(config, **overrides)


def describe(overrides: dict[str, str]) -> str:
    """A short suffix naming the overrides, for a result table's header line.

    Empty when there are none. Printed rather than assumed because a table whose
    header does not say which scenario it measured is a table that will later be
    compared against the wrong one.
    """
    if not overrides:
        return ""
    named = ", ".join(f"{key}={value}" for key, value in sorted(overrides.items()))
    return f"  [{named}]"
