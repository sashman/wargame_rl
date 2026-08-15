from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.types.game_timing import BattlePhase

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.env_components.distance_cache import DistanceCache


@dataclass(slots=True)
class StepContext:
    """Extensible data carrier assembled by the environment each step.

    Passed to all reward calculators and success criteria so their
    signatures stay stable as new mechanics (combat, terrain, VP) are
    added -- just add fields here.
    """

    distance_cache: DistanceCache
    current_turn: int
    max_turns: int
    board_width: int
    board_height: int
    is_terminated: bool = False
    current_round: int = 1
    battle_phase: BattlePhase = BattlePhase.command
    player_damage_dealt: int = 0
    opponent_damage_dealt: int = 0
    player_models_killed: int = 0
    opponent_models_killed: int = 0
    # Kills made by each player model this step, shape (n_player_models,).
    # `player_models_killed` is its sum; the vector exists so shooting can be
    # credited to the model that actually fired rather than shared flat across
    # the army. None when no shooting has been resolved this step.
    player_kills_by_model: np.ndarray | None = None
    # Per-model: did coherency enforcement move this model off the position it
    # chose this step? None outside the movement phase and when enforcement is
    # off. `coherency_intervention` is the only consumer.
    models_displaced_by_enforcement: np.ndarray | None = None
