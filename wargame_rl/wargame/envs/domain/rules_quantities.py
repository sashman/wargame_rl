"""Rules distances resolved into coordinate units, once, at construction.

Every rules quantity is authored in inches and has to be divided by the scale
before it can be compared against a coordinate. Doing that per call would put a
division in the hottest paths in the environment -- the shooting mask runs one
distance comparison per (shooter, target) pair per phase.

``RulesQuantities`` resolves them all once when the environment is built. Runtime
code reads plain floats off it and never converts. This is CLAUDE.md's "go from a
config to an execution context before usage": complexity at startup, simple at
runtime.

**Every value comes from the config, not from `rules_constants`.** That is
deliberate and is what makes introducing this module a no-op: the environment's
defaults are unchanged, so at the default scale of one inch per unit every
quantity resolves to exactly the number the code used before. Where the env's
default differs from the rules -- engagement range is 1", the rules say 2" --
the divergence is the config's to state and the gap map's to record. Adopting a
rules value is a *scenario change* that moves baselines, and belongs in its own
measured step.

This type holds only what is read through it. A quantity that no caller consumes
is not a placeholder for a future one; add it with the code that needs it, so a
config field can never be settable-but-inert.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.domain.scale import Scale

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.types.config import WargameEnvConfig


@dataclass(frozen=True, slots=True)
class RulesQuantities:
    """Rules distances in coordinate units, plus the scale they came from.

    Every distance field is in **units**. The scale is retained so callers that
    want to report in inches -- metrics, the renderer, the analysis layer -- can
    convert back without threading it separately.
    """

    scale: Scale
    engagement_range: float
    max_move_speed: float
    los_sample_step: float
    base_radius: float
    coherency_distance: float


def resolve_rules_quantities(config: WargameEnvConfig) -> RulesQuantities:
    """Resolve the config's rules distances into coordinate units.

    Called once per environment, at construction. At ``inches_per_unit: 1.0``
    the conversion is the identity, so this returns the config's own numbers.
    """
    scale = Scale(inches_per_unit=config.inches_per_unit)
    return RulesQuantities(
        scale=scale,
        engagement_range=scale.to_units(config.engagement_range),
        max_move_speed=scale.to_units(config.max_move_speed),
        los_sample_step=scale.to_units(config.los_sample_step),
        base_radius=scale.to_units(config.base_radius),
        coherency_distance=scale.to_units(config.group_max_distance),
    )
