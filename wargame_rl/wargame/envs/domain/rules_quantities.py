"""Rules distances resolved into coordinate units, once, at construction.

Every rules quantity is authored in inches and has to be divided by the scale before it
can be compared against a coordinate. Doing that per call would put a division in the
hottest paths in the environment -- the shooting mask runs one distance comparison per
(shooter, target) pair per phase.

``RulesQuantities`` resolves them all once when the environment is built. Runtime code
reads plain floats off it and never converts. This is the "go from a config to an
execution context before usage" rule in CLAUDE.md: complexity at startup, simple at
runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.domain import rules_constants
from wargame_rl.wargame.envs.domain.scale import Scale

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.types.config import WargameEnvConfig


@dataclass(frozen=True, slots=True)
class RulesQuantities:
    """Rules distances in coordinate units, plus the scale they came from.

    Every distance field is in **units**. The scale is retained so callers that want to
    report in inches -- metrics, the renderer, the analysis layer -- can convert back
    without threading it separately.
    """

    scale: Scale
    base_radius: float
    engagement_range: float
    objective_radius: float
    max_move_speed: float
    default_weapon_range: float
    group_max_distance: float
    coherency_nearest: float
    cover_ranged_skill_penalty: int

    @classmethod
    def resolve(
        cls,
        scale: Scale,
        *,
        base_radius_in: float | None = None,
        engagement_range_in: float | None = None,
        objective_radius_in: float | None = None,
        max_move_speed_in: float | None = None,
        default_weapon_range_in: float | None = None,
        group_max_distance_in: float | None = None,
    ) -> "RulesQuantities":
        """Resolve every rules distance into units under *scale*.

        Each keyword overrides the corresponding rules constant, in inches. ``None``
        takes the rules value. Overrides exist because a scenario may deliberately
        deviate -- a shorter weapon range makes terrain matter more, for instance --
        and that deviation should be visible in the config rather than buried here.
        """
        to_units = scale.to_units
        return cls(
            scale=scale,
            base_radius=to_units(
                _default(base_radius_in, rules_constants.MODEL_BASE_RADIUS_IN)
            ),
            engagement_range=to_units(
                _default(engagement_range_in, rules_constants.ENGAGEMENT_RANGE_IN)
            ),
            objective_radius=to_units(
                _default(objective_radius_in, rules_constants.OBJECTIVE_MARKER_RANGE_IN)
            ),
            max_move_speed=to_units(_default(max_move_speed_in, DEFAULT_MOVE_IN)),
            default_weapon_range=to_units(
                _default(default_weapon_range_in, DEFAULT_WEAPON_RANGE_IN)
            ),
            group_max_distance=to_units(
                _default(group_max_distance_in, rules_constants.COHERENCY_FURTHEST_IN)
            ),
            coherency_nearest=to_units(rules_constants.COHERENCY_NEAREST_IN),
            cover_ranged_skill_penalty=rules_constants.COVER_RANGED_SKILL_PENALTY,
        )


# Per-profile characteristics, not universal rules constants -- a model's Move and a
# weapon's Range are printed on its profile and vary between models. These are the
# defaults used when a config does not state one, chosen to match a typical infantry
# profile in the rules.
DEFAULT_MOVE_IN = 6.0
DEFAULT_WEAPON_RANGE_IN = 24.0


def resolve_rules_quantities(config: "WargameEnvConfig") -> RulesQuantities:
    """Resolve every rules distance in *config* from inches into coordinate units.

    Called once when the environment is built. Everything downstream reads units.
    """
    return RulesQuantities.resolve(
        Scale(inches_per_unit=config.inches_per_unit),
        base_radius_in=config.base_radius,
        engagement_range_in=config.engagement_range,
        objective_radius_in=config.objective_radius_size,
        max_move_speed_in=config.max_move_speed,
        group_max_distance_in=config.group_max_distance,
    )


def _default(override: float | None, fallback: float) -> float:
    """Return *override* when given, else *fallback*."""
    return fallback if override is None else override
