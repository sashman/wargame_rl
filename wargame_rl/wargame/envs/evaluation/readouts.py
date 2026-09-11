"""What the board says when an episode ends, read the same way by both facades.

Extracted verbatim from the tail of `evaluate_selector` so that `held`,
`on_obj`, `alive` and the cohesion gap have exactly one definition. It takes
the entity lists rather than an env, because the two facades are different
classes and this is a read over the aggregate's state, not over a facade.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wargame_rl.wargame.envs.domain.kernel.entities import (
    WargameModel,
    WargameObjective,
    alive_mask_for,
)
from wargame_rl.wargame.envs.env_components.distance_cache import compute_distances


@dataclass(frozen=True)
class EndOfEpisode:
    """The end-state readouts of one episode."""

    # Share of ALIVE player models standing inside any objective.
    at_objectives: float
    # Objectives the player controls: strictly more player than opponent
    # models inside the disc, the rule VP scores on; a tie scores for nobody.
    objectives_held: float
    fraction_alive: float
    # Largest distance from any alive model to its nearest living squadmate,
    # through the same helper `group_cohesion` uses, so it is comparable to a
    # phase's `group_max_distance`.
    worst_cohesion_gap: float


def read_end_of_episode(
    models: list[WargameModel],
    opponent_models: list[WargameModel],
    objectives: list[WargameObjective],
) -> EndOfEpisode:
    """Read the four end-state figures off the entity lists."""
    alive = alive_mask_for(models)
    cache = compute_distances(models, objectives, alive_mask=alive)
    at_objective = np.atleast_1d(
        (cache.model_obj_norms_offset <= cache.obj_radii).any(axis=1)
    )
    fraction_at = float(at_objective[alive].mean()) if alive.any() else 0.0

    opponent_alive = alive_mask_for(opponent_models)
    if opponent_models:
        opponent_norms = compute_distances(
            opponent_models, objectives, alive_mask=opponent_alive
        ).model_obj_norms_offset
        opponent_counts = (opponent_norms <= cache.obj_radii).sum(axis=0)
    else:
        opponent_counts = np.zeros(len(objectives), dtype=int)
    player_counts = (cache.model_obj_norms_offset[alive] <= cache.obj_radii).sum(axis=0)
    held = float((player_counts > opponent_counts).sum())

    return EndOfEpisode(
        at_objectives=fraction_at,
        objectives_held=held,
        fraction_alive=float(alive.mean()) if alive.size else 0.0,
        worst_cohesion_gap=worst_cohesion_gap(models, objectives),
    )


def worst_cohesion_gap(
    models: list[WargameModel], objectives: list[WargameObjective]
) -> float:
    """Largest distance from any alive model to its nearest living squadmate."""
    alive = alive_mask_for(models)
    if not alive.any():
        return 0.0
    cache = compute_distances(models, objectives, compute_model_model=True)
    group_ids = np.array([m.group_id for m in models], dtype=np.intp)
    distances = cache.min_distances_to_same_group(group_ids, alive_mask=alive)
    return float(distances[alive].max())


__all__ = ["EndOfEpisode", "read_end_of_episode", "worst_cohesion_gap"]
