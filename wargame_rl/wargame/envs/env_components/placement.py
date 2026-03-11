"""Backward-compat re-exports of placement from domain."""

from wargame_rl.wargame.envs.domain.placement import (
    fixed_objective_placement,
    fixed_wargame_model_placement,
    objective_placement,
    place_for_episode,
    wargame_model_placement,
)

__all__ = [
    "fixed_objective_placement",
    "fixed_wargame_model_placement",
    "objective_placement",
    "place_for_episode",
    "wargame_model_placement",
]
