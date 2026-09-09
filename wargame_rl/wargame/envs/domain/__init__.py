"""The battle domain: one bounded context, shaped as sub-domains.

`kernel/` is the shared kernel every sub-domain uses; `battle.py`,
`battle_factory.py` and `battle_view.py` are the aggregate, its factory and its
read contract; `battlefield/`, `sequencing/`, `movement/`, `attacks/`,
`shooting/` and `melee/` are the sub-domains, each a chapter or two of
`docs/rules/`. The names re-exported here are the ones the application layer
reaches for by habit; everything else is imported from its sub-domain.
"""

from wargame_rl.wargame.envs.domain.attacks.sequence import wound_roll_threshold
from wargame_rl.wargame.envs.domain.attacks.stats import (
    DefenderStats,
    ShootingResult,
    WeaponStats,
)
from wargame_rl.wargame.envs.domain.battle import Battle
from wargame_rl.wargame.envs.domain.battle_factory import (
    create_objectives,
    create_opponent_models,
    create_wargame_models,
    from_config,
)
from wargame_rl.wargame.envs.domain.battle_view import BattleView
from wargame_rl.wargame.envs.domain.battlefield.los import segments_are_clear
from wargame_rl.wargame.envs.domain.battlefield.placement import place_for_episode
from wargame_rl.wargame.envs.domain.battlefield.sight import (
    has_line_of_sight_between_points,
    line_of_sight_matrix,
)
from wargame_rl.wargame.envs.domain.kernel.entities import (
    WargameModel,
    WargameObjective,
)
from wargame_rl.wargame.envs.domain.kernel.value_objects import (
    BoardDimensions,
    DeploymentZone,
)
from wargame_rl.wargame.envs.domain.sequencing.game_clock import (
    GameClock,
    GameClockError,
)
from wargame_rl.wargame.envs.domain.sequencing.termination import (
    check_max_turns_reached,
    is_battle_over,
)
from wargame_rl.wargame.envs.domain.shooting.expectation import expected_damage
from wargame_rl.wargame.envs.domain.shooting.resolve import (
    resolve_shooting,
    resolve_shooting_phase,
)

__all__ = [
    "Battle",
    "BoardDimensions",
    "BattleView",
    "segments_are_clear",
    "has_line_of_sight_between_points",
    "line_of_sight_matrix",
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
    "DefenderStats",
    "ShootingResult",
    "WeaponStats",
    "expected_damage",
    "resolve_shooting",
    "resolve_shooting_phase",
    "wound_roll_threshold",
]
