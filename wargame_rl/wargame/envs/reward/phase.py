from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SuccessCriteriaConfig(BaseModel):
    """YAML-serialisable description of a success criteria."""

    type: str = Field(
        description="Registry key, e.g. 'all_at_objectives', 'all_models_grouped'"
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments forwarded to the criteria constructor",
    )


class RewardCalculatorConfig(BaseModel):
    """YAML-serialisable description of a single reward calculator."""

    type: str = Field(
        description="Registry key, e.g. 'closest_objective', 'group_cohesion'"
    )
    weight: float = Field(
        default=1.0, description="Multiplier applied to this calculator's output"
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments forwarded to the calculator constructor",
    )


class RewardPhaseConfig(BaseModel):
    """Configuration for a single reward phase in the curriculum."""

    name: str = Field(description="Human-readable phase name")
    reward_calculators: list[RewardCalculatorConfig] = Field(
        min_length=1,
        description="Reward calculators active during this phase",
    )
    success_criteria: SuccessCriteriaConfig = Field(
        description="Criteria that determines phase success for an episode"
    )
    success_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Fraction of eval episodes that must succeed to advance",
    )
    min_epochs: int = Field(
        default=0,
        ge=0,
        description="Minimum epochs in this phase before eligible to advance",
    )
