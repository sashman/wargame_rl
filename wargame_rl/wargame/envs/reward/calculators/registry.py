from __future__ import annotations

from typing import Any

from wargame_rl.wargame.envs.reward.calculators.base import (
    GlobalRewardCalculator,
    PerModelRewardCalculator,
)
from wargame_rl.wargame.envs.reward.calculators.charge_progress import (
    ChargeProgressCalculator,
)
from wargame_rl.wargame.envs.reward.calculators.closest_objective import (
    ClosestObjectiveCalculator,
)
from wargame_rl.wargame.envs.reward.calculators.closest_objective_v2 import (
    ClosestObjectiveV2Calculator,
)
from wargame_rl.wargame.envs.reward.calculators.group_cohesion import (
    GroupCohesionCalculator,
)
from wargame_rl.wargame.envs.reward.calculators.killing import KillingReward
from wargame_rl.wargame.envs.reward.calculators.model_kills import ModelKillsCalculator
from wargame_rl.wargame.envs.reward.calculators.models_at_objectives import (
    ModelsAtObjectivesCalculator,
)
from wargame_rl.wargame.envs.reward.calculators.models_lost import ModelsLostPenalty
from wargame_rl.wargame.envs.reward.calculators.objective_coverage import (
    ObjectiveCoverageCalculator,
)
from wargame_rl.wargame.envs.reward.calculators.objective_flip_bonus import (
    ObjectiveFlipBonusCalculator,
)
from wargame_rl.wargame.envs.reward.calculators.objective_hold import (
    ObjectiveHoldCalculator,
)
from wargame_rl.wargame.envs.reward.calculators.unit_coherency import (
    UnitCoherencyCalculator,
)
from wargame_rl.wargame.envs.reward.calculators.vp_gain import VPGainCalculator

RewardCalculatorType = PerModelRewardCalculator | GlobalRewardCalculator

CALCULATOR_REGISTRY: dict[str, type[RewardCalculatorType]] = {
    "closest_objective": ClosestObjectiveCalculator,
    "closest_objective_v2": ClosestObjectiveV2Calculator,
    "group_cohesion": GroupCohesionCalculator,
    "model_kills": ModelKillsCalculator,
    "models_at_objectives": ModelsAtObjectivesCalculator,
    "models_lost": ModelsLostPenalty,
    "objective_coverage": ObjectiveCoverageCalculator,
    "objective_flip_bonus": ObjectiveFlipBonusCalculator,
    "objective_hold": ObjectiveHoldCalculator,
    "charge_progress": ChargeProgressCalculator,
    "unit_coherency": UnitCoherencyCalculator,
    "vp_gain": VPGainCalculator,
    "killing": KillingReward,
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
