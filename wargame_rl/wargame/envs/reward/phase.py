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
    min_epochs_above_threshold: int = Field(
        default=5,
        ge=0,
        description="Success rate must be >= success_threshold for this many consecutive epochs before advancing.",
    )
    terminal_success_bonus: float = Field(
        default=0.0,
        description="Bonus added at episode end when all models are at an objective. "
        "Scaled by remaining turns fraction. 0 disables.",
    )
    terminal_vp_bonus: float = Field(
        default=0.0,
        description="Bonus added at episode end when player VP meets the phase's "
        "VP threshold. 0 disables.",
    )
    terminate_on_success: bool = Field(
        default=True,
        description="If True, episode ends early when all models reach an objective. "
        "Set to False to let the episode run to the turn limit.",
    )
