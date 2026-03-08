from __future__ import annotations

from typing import Any

from wargame_rl.wargame.envs.reward.calculators.base import (
    GlobalRewardCalculator,
    PerModelRewardCalculator,
)
from wargame_rl.wargame.envs.reward.calculators.closest_objective import (
    ClosestObjectiveCalculator,
)
from wargame_rl.wargame.envs.reward.calculators.group_cohesion import (
    GroupCohesionCalculator,
)
from wargame_rl.wargame.envs.reward.calculators.objective_control import (
    ObjectiveControlCalculator,
)

RewardCalculatorType = PerModelRewardCalculator | GlobalRewardCalculator

CALCULATOR_REGISTRY: dict[str, type[RewardCalculatorType]] = {
    "closest_objective": ClosestObjectiveCalculator,
    "group_cohesion": GroupCohesionCalculator,
    "objective_control": ObjectiveControlCalculator,
}


def build_calculator(
    type_name: str, weight: float, params: dict[str, Any]
) -> RewardCalculatorType:
    """Instantiate a reward calculator by its registry name."""
    cls = CALCULATOR_REGISTRY.get(type_name)
    if cls is None:
        available = ", ".join(sorted(CALCULATOR_REGISTRY.keys()))
        raise ValueError(
            f"Unknown reward calculator type '{type_name}'. Available: {available}"
        )
    return cls(weight=weight, **params)
