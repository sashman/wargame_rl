"""Modular components for the wargame environment (actions, termination, placement, observation)."""

from wargame_rl.wargame.envs.env_components.actions import ActionHandler
from wargame_rl.wargame.envs.env_components.distance_cache import (
    DistanceCache,
    compute_distances,
)
from wargame_rl.wargame.envs.env_components.game_clock import GameClock, GameClockError
from wargame_rl.wargame.envs.env_components.observation_builder import (
    build_info,
    build_observation,
    update_distances_to_objectives,
)
from wargame_rl.wargame.envs.env_components.placement import (
    fixed_objective_placement,
    fixed_wargame_model_placement,
    objective_placement,
    wargame_model_placement,
)
from wargame_rl.wargame.envs.env_components.shooting_masks import (
    compute_shooting_masks,
    max_weapon_ranges,
)
from wargame_rl.wargame.envs.env_components.termination import (
    check_max_turns_reached,
    get_termination,
)

__all__ = [
    "ActionHandler",
    "DistanceCache",
    "GameClock",
    "GameClockError",
    "build_info",
    "build_observation",
    "compute_distances",
    "compute_shooting_masks",
    "max_weapon_ranges",
    "update_distances_to_objectives",
    "wargame_model_placement",
    "objective_placement",
    "fixed_wargame_model_placement",
    "fixed_objective_placement",
    "get_termination",
    "check_max_turns_reached",
]
