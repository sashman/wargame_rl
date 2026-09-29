from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SuccessCriteriaConfig(BaseModel):
    """YAML-serialisable description of a success criteria."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(
        description="Registry key, e.g. 'all_at_objectives', 'all_models_grouped'"
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments forwarded to the criteria constructor",
    )


class RewardCalculatorConfig(BaseModel):
    """YAML-serialisable description of a single reward calculator."""

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

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
        description="Bonus added at episode end when the phase's success_criteria is "
        "met. Scaled by remaining turns fraction (faster success = higher). 0 disables.",
    )
    terminal_bonus_speed_scaling: bool = Field(
        default=True,
        description="Scale `terminal_success_bonus` by the fraction of turns left "
        "when success ends the episode (a speed incentive). False pays the bonus "
        "in full whenever the criteria hold, so a late success is worth as much "
        "as an early one; the speed is then read from `turns`, not paid.",
    )
    terminal_member_success_bonus: float = Field(
        default=0.0,
        ge=0.0,
        description="The SOLDIERS' success bonus (#384, 2026-09-27; the per-model "
        "facade only): the soldiers' most pay per round, summed over the army "
        "(plan_following's weight at a score of 1). Paid to the members' stream "
        "at a success that ends the game as that times sum_{m=1..R} gamma^m, R "
        "the rounds of play the success cuts off and gamma the members' "
        "discount per round -- exactly the most the soldiers forgo, so "
        "completing the plan never costs them and never pays them for "
        "abandoning it. 0 disables.",
    )
    terminal_forgone_vp_bonus: float = Field(
        default=0.0,
        ge=0.0,
        description="The PLANNER's success bonus in VP (#384, 2026-09-27; the "
        "per-model facade only): the planner's most pay per VP scoring round "
        "(vp_gain's weight at the whole board held). Paid to the planning "
        "stream at a success that ends the game as that times (1 + "
        "`terminal_forgone_vp_margin`) times sum_{m=1..S} gamma_p^m, S the VP "
        "scoring rounds the success cuts off and gamma_p the planning "
        "discount -- slightly more than the most VP the planner forgoes, so "
        "finishing is strictly its best. 0 disables.",
    )
    terminal_forgone_vp_margin: float = Field(
        default=0.0,
        ge=0.0,
        description="The planner's advantage for finishing: the share by which "
        "`terminal_forgone_vp_bonus` exceeds the VP a success forgoes.",
    )
    terminal_objective_bonus: float = Field(
        default=0.0,
        description="Bonus added at episode end scaled by the fraction of "
        "objectives the player controls on the final board, under VP's control "
        "rule, whether or not the phase's success_criteria are met. Never scaled "
        "by the turns left. 0 disables.",
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
