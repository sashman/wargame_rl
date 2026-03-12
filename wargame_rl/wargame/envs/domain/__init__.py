"""Domain layer: battle aggregate, entities, view protocol, and services."""

from wargame_rl.wargame.envs.domain.battle import Battle
from wargame_rl.wargame.envs.domain.battle_factory import (
    create_objectives,
    create_opponent_models,
    create_wargame_models,
    from_config,
)
from wargame_rl.wargame.envs.domain.battle_view import BattleView
from wargame_rl.wargame.envs.domain.entities import WargameModel, WargameObjective
from wargame_rl.wargame.envs.domain.game_clock import GameClock, GameClockError
from wargame_rl.wargame.envs.domain.placement import place_for_episode
from wargame_rl.wargame.envs.domain.termination import (
    check_max_turns_reached,
    is_battle_over,
)
from wargame_rl.wargame.envs.domain.turn_execution import (
    run_after_player_action,
    run_until_player_phase,
)
from wargame_rl.wargame.envs.domain.value_objects import BoardDimensions, DeploymentZone

__all__ = [
    "Battle",
    "BoardDimensions",
    "BattleView",
    "WargameModel",
    "WargameObjective",
    "DeploymentZone",
    "GameClock",
    "GameClockError",
    "place_for_episode",
    "check_max_turns_reached",
    "is_battle_over",
    "create_objectives",
    "create_opponent_models",
    "create_wargame_models",
    "from_config",
    "run_after_player_action",
    "run_until_player_phase",
]
