"""Turn order, opponent policy and mission selection."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TurnOrder(str, Enum):
    """Who moves first each turn."""

    player = "player"
    opponent = "opponent"
    random = "random"


class OpponentPolicyConfig(BaseModel):
    """Configuration for the opponent policy engine."""

    type: str = Field(
        description="Policy engine identifier, e.g. 'random', 'scripted_advance_to_objective'."
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Policy-specific parameters forwarded to the policy constructor.",
    )


class MissionConfig(BaseModel):
    """Configuration for the mission (victory point scoring rules)."""

    type: str = Field(
        default="default",
        description="Mission type identifier; selects the VP calculator (e.g. 'default', 'none').",
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Mission-specific parameters (e.g. vp_per_objective, cap_per_turn, min_round).",
    )
