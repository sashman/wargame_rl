"""Pin the domain to docs/rules/constants.yaml wherever the env claims fidelity.

`docs/rules/constants.yaml` is the rules' source of truth. Most of it describes rules
the environment simplifies or has not implemented; those divergences are documented in
`docs/rules/implementation-status.md` rather than asserted here. What *is* asserted is
the small set of rules the environment implements faithfully today, so that a change to
either the rules file or the domain cannot silently pull them apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from wargame_rl.wargame.envs.domain.shooting import wound_roll_threshold

CONSTANTS_PATH = (
    Path(__file__).resolve().parents[1] / "docs" / "rules" / "constants.yaml"
)


@pytest.fixture(scope="module")
def constants() -> dict[str, Any]:
    """The rules constants, parsed once per module."""
    with CONSTANTS_PATH.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    assert isinstance(loaded, dict)
    return loaded


def test_constants_file_exists_and_parses(constants: dict[str, Any]) -> None:
    """The rules reference ships a parseable constants file."""
    assert constants["battle"]["rounds"] == 5
    assert constants["battlefield"]["width_in"] == 44


# (strength, toughness) -> the key in constants.yaml:wound_ladder that governs it.
WOUND_LADDER_CASES = [
    (8, 4, "s_at_least_double_t"),
    (10, 3, "s_at_least_double_t"),
    (5, 4, "s_greater_than_t"),
    (4, 4, "s_equal_t"),
    (3, 4, "s_less_than_t"),
    (2, 4, "s_at_most_half_t"),
    (1, 5, "s_at_most_half_t"),
]


@pytest.mark.parametrize(("strength", "toughness", "rung"), WOUND_LADDER_CASES)
def test_wound_ladder_matches_the_rules(
    constants: dict[str, Any], strength: int, toughness: int, rung: str
) -> None:
    """`wound_roll_threshold` reproduces the documented Strength-vs-Toughness ladder."""
    expected = constants["wound_ladder"][rung]

    assert wound_roll_threshold(strength, toughness) == expected


def test_wound_ladder_covers_every_documented_rung(constants: dict[str, Any]) -> None:
    """Every rung in the rules file is exercised, so a new rung cannot go untested."""
    exercised = {rung for _, _, rung in WOUND_LADDER_CASES}

    assert exercised == set(constants["wound_ladder"])


def test_wound_ladder_never_leaves_the_documented_range(
    constants: dict[str, Any],
) -> None:
    """No Strength/Toughness pairing produces a threshold outside the ladder."""
    thresholds = set(constants["wound_ladder"].values())

    produced = {
        wound_roll_threshold(strength, toughness)
        for strength in range(1, 21)
        for toughness in range(1, 21)
    }

    assert produced <= thresholds


def test_critical_hit_and_wound_results_are_the_documented_ones(
    constants: dict[str, Any],
) -> None:
    """The attack sequence keys criticals off an unmodified 6 on both rolls."""
    assert constants["attack"]["critical_hit_on"] == 6
    assert constants["attack"]["critical_wound_on"] == 6
    assert constants["attack"]["unmodified_1_always_fails"] is True
