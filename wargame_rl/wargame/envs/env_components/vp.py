"""Victory Points (VP) calculation and state for primary mission scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.wargame_model import WargameModel
    from wargame_rl.wargame.envs.wargame_objective import WargameObjective

VP_PER_OBJECTIVE = 5
VP_CAP_PER_TURN = 15


@dataclass
class VPState:
    """Holds cumulative VP and per-step VP gains for player and opponent."""

    player_vp: int = 0
    opponent_vp: int = 0
    vp_gained_this_step_player: int = 0
    vp_gained_this_step_opponent: int = 0

    def reset(self) -> None:
        """Clear all VP and step gains (e.g. at episode reset)."""
        self.player_vp = 0
        self.opponent_vp = 0
        self.vp_gained_this_step_player = 0
        self.vp_gained_this_step_opponent = 0

    def start_step(self) -> None:
        """Clear per-step gains at the beginning of a step."""
        self.vp_gained_this_step_player = 0
        self.vp_gained_this_step_opponent = 0

    def award_player(self, vp: int) -> None:
        """Add VP to player total and record as this step's player gain."""
        self.player_vp += vp
        self.vp_gained_this_step_player = vp

    def award_opponent(self, vp: int) -> None:
        """Add VP to opponent total and record as this step's opponent gain."""
        self.opponent_vp += vp
        self.vp_gained_this_step_opponent += vp


def compute_primary_vp_earned(
    player_models: list["WargameModel"],
    opponent_models: list["WargameModel"],
    objectives: list["WargameObjective"],
    control_range: float,
) -> tuple[int, int]:
    """VP earned this scoring moment (5 per objective controlled, cap 15).

    Returns (player_vp, opponent_vp) for the given board state.
    """
    from wargame_rl.wargame.envs.env_components.distance_cache import (
        compute_levels_of_control,
    )

    player_loc, opponent_loc = compute_levels_of_control(
        player_models,
        opponent_models,
        objectives,
        control_range,
    )
    player_controlled = int((player_loc > opponent_loc).sum())
    opponent_controlled = int((opponent_loc > player_loc).sum())
    return (
        min(VP_PER_OBJECTIVE * player_controlled, VP_CAP_PER_TURN),
        min(VP_PER_OBJECTIVE * opponent_controlled, VP_CAP_PER_TURN),
    )
