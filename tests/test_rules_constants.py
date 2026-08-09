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

from wargame_rl.wargame.envs.domain import rules_constants
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


def test_runtime_constants_match_the_specification(constants: dict[str, Any]) -> None:
    """`domain/rules_constants.py` is the runtime mirror of this YAML.

    The YAML is documentation and is never read at runtime -- no file IO on a hot
    path, and no packaging problem shipping a data file. The cost of that choice
    is two copies of every number, so this test is what stops them drifting.

    A value appears in the runtime module only once something reads it. Adding a
    constant there without adding it here is how the two quietly diverge.
    """
    assert (
        rules_constants.ENGAGEMENT_RANGE_IN == constants["engagement"]["horizontal_in"]
    )
    assert rules_constants.COHERENCY_NEAREST_IN == constants["coherency"]["nearest_in"]
    assert (
        rules_constants.COHERENCY_FURTHEST_IN == constants["coherency"]["furthest_in"]
    )
    assert (
        rules_constants.MODEL_BASE_DIAMETER_MM
        == constants["models"]["base_diameter_mm"]
    )
    assert (
        rules_constants.OBJECTIVE_MARKER_RANGE_IN
        == constants["objectives"]["marker_range_in"]
    )
    assert (
        rules_constants.COVER_RANGED_SKILL_PENALTY
        == constants["cover"]["ranged_skill_penalty"]
    )
    assert rules_constants.BATTLE_ROUNDS == constants["battle"]["rounds"]


def test_base_radius_is_half_the_diameter_in_inches() -> None:
    """The one derived constant, so the derivation cannot rot silently."""
    expected = rules_constants.MODEL_BASE_DIAMETER_MM / rules_constants.MM_PER_INCH / 2
    assert rules_constants.MODEL_BASE_RADIUS_IN == expected
    assert 0.6 < rules_constants.MODEL_BASE_RADIUS_IN < 0.65
