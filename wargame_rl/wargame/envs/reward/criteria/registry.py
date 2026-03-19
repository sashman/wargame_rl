from __future__ import annotations

from typing import Any

from wargame_rl.wargame.envs.reward.criteria.all_at_objectives import (
    AllAtObjectivesCriteria,
)
from wargame_rl.wargame.envs.reward.criteria.all_models_grouped import (
    AllModelsGroupedCriteria,
)
from wargame_rl.wargame.envs.reward.criteria.base import SuccessCriteria
from wargame_rl.wargame.envs.reward.criteria.player_vp_min import PlayerVPMinCriteria

CRITERIA_REGISTRY: dict[str, type[SuccessCriteria]] = {
    "all_at_objectives": AllAtObjectivesCriteria,
    "all_models_grouped": AllModelsGroupedCriteria,
    "player_vp_min": PlayerVPMinCriteria,
}


def build_criteria(type_name: str, params: dict[str, Any]) -> SuccessCriteria:
    """Instantiate a success criteria by its registry name."""
    cls = CRITERIA_REGISTRY.get(type_name)
    if cls is None:
        available = ", ".join(sorted(CRITERIA_REGISTRY.keys()))
        raise ValueError(
            f"Unknown success criteria type '{type_name}'. Available: {available}"
        )
    return cls(**params)
