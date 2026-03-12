"""Termination logic for the wargame environment.

Re-exports from domain and provides get_termination(distance_cache) for
callers that have a DistanceCache.  Battle-over logic is in domain.termination.
"""

from __future__ import annotations

from wargame_rl.wargame.envs.domain.termination import check_max_turns_reached
from wargame_rl.wargame.envs.env_components.distance_cache import DistanceCache


def get_termination(distance_cache: DistanceCache) -> bool:
    """True if every model has reached at least one objective."""
    return distance_cache.all_models_at_objectives()


__all__ = ["check_max_turns_reached", "get_termination"]
