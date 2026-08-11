"""The inches-to-units scale, and the quantities resolved through it."""

from __future__ import annotations

import pytest

from wargame_rl.wargame.envs.domain.rules_quantities import (
    RulesQuantities,
    resolve_rules_quantities,
)
from wargame_rl.wargame.envs.domain.scale import Scale
from wargame_rl.wargame.envs.env_components.actions import ActionHandler
from wargame_rl.wargame.envs.types.config import WargameEnvConfig


class TestScale:
    """Rules distances are authored in inches; coordinates are in units."""

    def test_the_default_scale_is_the_identity(self) -> None:
        """The property the whole change rests on.

        At one inch per unit every conversion is a no-op, which is what let the
        scale be introduced without moving a single measured result.
        """
        scale = Scale()

        assert scale.inches_per_unit == 1.0
        assert scale.to_units(12.0) == 12.0
        assert scale.to_inches(12.0) == 12.0

    @pytest.mark.parametrize("inches_per_unit", [0.5, 2.0, 4.0])
    def test_conversion_round_trips(self, inches_per_unit: float) -> None:
        scale = Scale(inches_per_unit=inches_per_unit)

        assert scale.to_inches(scale.to_units(24.0)) == pytest.approx(24.0)

    def test_a_finer_scale_makes_a_rules_distance_span_more_units(self) -> None:
        """Half an inch per unit means 12 inches covers 24 units, not 6."""
        assert Scale(inches_per_unit=0.5).to_units(12.0) == 24.0

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_a_non_positive_scale_is_rejected_at_construction(self, bad: float) -> None:
        """Zero would make every conversion a division by zero at runtime."""
        with pytest.raises(ValueError, match="inches_per_unit must be positive"):
            Scale(inches_per_unit=bad)


class TestRulesQuantities:
    """Resolved once from config, in units, never converted again at runtime."""

    def test_defaults_resolve_to_the_values_the_code_used_before(self) -> None:
        """The no-op guarantee, pinned.

        Every quantity here was a config field or a module constant with these
        exact values before the scale existed. If one of them moves, every
        baseline in the repo was measured under a different environment.
        """
        quantities = resolve_rules_quantities(WargameEnvConfig())

        assert quantities.scale.inches_per_unit == 1.0
        assert quantities.engagement_range == 1.0
        assert quantities.max_move_speed == 6.0

    def test_the_scale_applies_to_every_distance(self) -> None:
        """Halving the unit doubles how many units a rules distance spans."""
        config = WargameEnvConfig(inches_per_unit=0.5)

        quantities = resolve_rules_quantities(config)

        assert quantities.engagement_range == 2.0
        assert quantities.max_move_speed == 12.0

    def test_it_is_frozen(self) -> None:
        """Resolved once at construction; nothing may edit it mid-episode."""
        quantities = resolve_rules_quantities(WargameEnvConfig())

        with pytest.raises(AttributeError):
            quantities.engagement_range = 5.0  # type: ignore[misc]

    def test_it_holds_only_what_is_read_through_it(self) -> None:
        """A quantity nobody consumes is a config field that silently does nothing.

        This project has repeatedly paid for levers the agent could not act on,
        so the type is deliberately small and grows with its callers rather than
        ahead of them. Update this list when a work package adds a consumer.
        """
        assert set(RulesQuantities.__dataclass_fields__) == {
            "scale",
            "engagement_range",
            "max_move_speed",
            "los_sample_step",
            "base_radius",
            "coherency_distance",
        }


def test_movement_distance_follows_the_scale() -> None:
    """`ActionHandler` sizes its displacements from the resolved move allowance.

    Reading `max_move_speed` straight off the config would have been correct
    only while one unit is one inch, and wrong silently otherwise -- models
    would move the right *number* on a board where the number means something
    else.
    """
    default_handler = ActionHandler(WargameEnvConfig(), n_models=2)
    finer_handler = ActionHandler(WargameEnvConfig(inches_per_unit=0.5), n_models=2)

    assert finer_handler._speeds.max() == 2 * default_handler._speeds.max()


class TestCoherencyDistanceIsOneNumber:
    """Placement and `group_cohesion` must enforce the same coherency distance.

    They did not. `group_max_distance` on the env config drove *placement* and
    defaulted to 10.0, while every shipped config set `group_cohesion`'s own
    `group_max_distance` to 6.0 — so squads spawned scattered to 10 and were
    fined from 6, and 199 of 200 episodes began in violation of a rule nobody
    had chosen to break. The config field's own description used to state the
    split as intentional.
    """

    def test_the_calculator_defaults_to_the_scenario_distance(self) -> None:
        """Unset, the term fines from exactly where placement stops spawning."""
        # Arrange
        config = WargameEnvConfig(group_max_distance=6.0)

        # Act
        quantities = resolve_rules_quantities(config)

        # Assert
        assert quantities.coherency_distance == 6.0

    def test_it_scales_like_every_other_rules_distance(self) -> None:
        """Authored in inches, resolved into units once, like the rest."""
        # Arrange
        config = WargameEnvConfig(group_max_distance=6.0, inches_per_unit=0.5)

        # Act
        quantities = resolve_rules_quantities(config)

        # Assert
        assert quantities.coherency_distance == 12.0
