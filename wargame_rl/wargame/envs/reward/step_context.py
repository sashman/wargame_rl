from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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
    current_round: int = 1
    battle_phase: BattlePhase = BattlePhase.command
