"""Termination logic for the wargame environment.

Extracted so termination conditions (all-at-objective, max turns, custom rules)
can be extended or swapped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.env_components.distance_cache import DistanceCache


def get_termination(
    current_turn: int,
    max_turns: int,
    distance_cache: DistanceCache,
) -> bool:
    """True if every model has reached at least one objective."""

    if check_max_turns_reached(current_turn, max_turns):
        return True

    # model_obj_norms_offset[i, j] <= obj_radii[j] means model i is at objective j
    at_objective = distance_cache.model_obj_norms_offset <= distance_cache.obj_radii
    return bool(at_objective.any(axis=1).all())


def check_max_turns_reached(current_turn: int, max_turns: int) -> bool:
    """True if the turn limit has been reached."""
    return current_turn >= max_turns
