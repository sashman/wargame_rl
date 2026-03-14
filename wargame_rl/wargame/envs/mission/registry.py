"""Registry for mission VP calculators."""

from __future__ import annotations

from typing import Any

from wargame_rl.wargame.envs.mission.vp_calculator import (
    DefaultVPCalculator,
    NoneVPCalculator,
    VPCalculator,
)

VP_CALCULATOR_REGISTRY: dict[str, type[VPCalculator]] = {
    "default": DefaultVPCalculator,
    "none": NoneVPCalculator,
}


def build_vp_calculator(type_name: str, params: dict[str, Any]) -> VPCalculator:
    """Instantiate a VP calculator by registry name."""
    cls = VP_CALCULATOR_REGISTRY.get(type_name)
    if cls is None:
        available = ", ".join(sorted(VP_CALCULATOR_REGISTRY.keys()))
        raise ValueError(
            f"Unknown mission VP calculator type '{type_name}'. Available: {available}"
        )
    return cls(**params)
