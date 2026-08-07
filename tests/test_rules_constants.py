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
from wargame_rl.wargame.envs.domain.rules_quantities import RulesQuantities
from wargame_rl.wargame.envs.domain.scale import Scale
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


# Each runtime constant in domain/rules_constants.py, and the path to the value it
# mirrors in docs/rules/constants.yaml. The runtime module is what the environment
# reads; the YAML is the specification. Neither may drift from the other.
MIRRORED_CONSTANTS = [
    ("ENGAGEMENT_RANGE_IN", ("engagement", "horizontal_in")),
    ("COHERENCY_NEAREST_IN", ("coherency", "nearest_in")),
    ("COHERENCY_FURTHEST_IN", ("coherency", "furthest_in")),
    ("MODEL_BASE_DIAMETER_MM", ("models", "base_diameter_mm")),
    ("COVER_RANGED_SKILL_PENALTY", ("cover", "ranged_skill_penalty")),
    ("DETECTION_RANGE_IN", ("visibility", "detection_range_in")),
    ("OBJECTIVE_MARKER_RANGE_IN", ("objectives", "marker_range_in")),
    ("BATTLE_ROUNDS", ("battle", "rounds")),
]


@pytest.mark.parametrize(("name", "path"), MIRRORED_CONSTANTS)
def test_runtime_constant_matches_the_specification(
    constants: dict[str, Any], name: str, path: tuple[str, ...]
) -> None:
    """Every runtime rules constant equals the value documented for it."""
    expected: Any = constants
    for key in path:
        expected = expected[key]

    assert getattr(rules_constants, name) == expected


def test_base_radius_is_half_the_base_diameter() -> None:
    """The base radius is derived, not authored, so it cannot disagree with the size."""
    assert rules_constants.MODEL_BASE_RADIUS_IN == pytest.approx(
        rules_constants.MODEL_BASE_DIAMETER_MM / 25.4 / 2.0
    )
    # A 32mm base is a little over an inch and a quarter across.
    assert rules_constants.MODEL_BASE_RADIUS_IN == pytest.approx(0.6299, abs=1e-4)


def test_default_scale_makes_units_and_inches_coincide() -> None:
    """At one inch per unit every rules distance passes through unchanged."""
    quantities = RulesQuantities.resolve(Scale())

    assert quantities.engagement_range == rules_constants.ENGAGEMENT_RANGE_IN
    assert quantities.objective_radius == rules_constants.OBJECTIVE_MARKER_RANGE_IN
    assert quantities.group_max_distance == rules_constants.COHERENCY_FURTHEST_IN


def test_scale_converts_every_distance_together() -> None:
    """Halving the inches per unit doubles every distance in units, uniformly."""
    baseline = RulesQuantities.resolve(Scale(inches_per_unit=1.0))
    halved = RulesQuantities.resolve(Scale(inches_per_unit=0.5))

    assert halved.base_radius == pytest.approx(2 * baseline.base_radius)
    assert halved.engagement_range == pytest.approx(2 * baseline.engagement_range)
    assert halved.objective_radius == pytest.approx(2 * baseline.objective_radius)
    assert halved.max_move_speed == pytest.approx(2 * baseline.max_move_speed)
    assert halved.default_weapon_range == pytest.approx(
        2 * baseline.default_weapon_range
    )


def test_overrides_replace_the_rules_value() -> None:
    """A scenario that deviates does so visibly, through an override in inches."""
    quantities = RulesQuantities.resolve(Scale(), default_weapon_range_in=12.0)

    assert quantities.default_weapon_range == 12.0
    # An override touches only what it names.
    assert quantities.engagement_range == rules_constants.ENGAGEMENT_RANGE_IN


def test_non_positive_scale_is_rejected() -> None:
    """A zero or negative scale would make every conversion meaningless."""
    with pytest.raises(ValueError, match="inches_per_unit"):
        Scale(inches_per_unit=0.0)
