"""Turn order, opponent policy and mission selection."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TurnOrder(str, Enum):
    """Who moves first each turn."""

    player = "player"
    opponent = "opponent"
    random = "random"


class OpponentPolicyConfig(BaseModel):
    """Configuration for the opponent policy engine."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(
        description="Policy engine identifier, e.g. 'random', 'scripted_advance_to_objective'."
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Policy-specific parameters forwarded to the policy constructor.",
    )


class MissionConfig(BaseModel):
    """Configuration for the mission (victory point scoring rules)."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(
        default="default",
        description="Mission type identifier; selects the VP calculator (e.g. 'default', 'none').",
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Mission-specific parameters (e.g. vp_per_objective, cap_per_turn, min_round).",
    )

    # The three numbers other layers need from a mission, as properties rather
    # than dict lookups on `params`.
    #
    # ⚠ THIS EXISTS BECAUSE THE STRING `"default"` WAS LOAD-BEARING AND SILENT.
    # `reward/criteria/player_vp_min.py` returned a theoretical max of **0** for
    # any `mission.type != "default"`, so the phase-advance threshold collapsed
    # to `min_vp` and `success_rate` pinned at 1.0 -- a curriculum advancing on
    # epoch count alone, with nothing logged to say so. On the same branch
    # `vp_threshold_for_terminal_bonus` returned None, disabling the terminal
    # bonus. `reward/calculators/vp_gain.py` divided by a cap it read through
    # three `getattr`s and silently fell back to 15.
    #
    # Every one of the 115 configs in this repo leaves `mission` at its default,
    # so nothing has ever exercised those branches -- the first config to set a
    # mission would have hit all three at once, and none of them fails loudly.

    @property
    def points_per_objective(self) -> int:
        """VP each controlled objective pays, before any cap."""
        return int(self.params.get("vp_per_objective", 5))

    @property
    def per_round_cap(self) -> int:
        """Most VP one side can score in a single scoring event."""
        return int(self.params.get("cap_per_turn", 15))

    @property
    def first_scoring_round(self) -> int:
        """First battle round in which VP are awarded."""
        return int(self.params.get("min_round", 2))
