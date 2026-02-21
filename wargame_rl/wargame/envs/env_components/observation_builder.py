"""Build observations and info from current env state.

Extracted so observation shape or content can be varied without touching step/reset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.types import (
    WargameEnvInfo,
    WargameEnvObjectiveObservation,
    WargameEnvObservation,
    WargameModelObservation,
)

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.env_components.distance_cache import DistanceCache
    from wargame_rl.wargame.envs.wargame_model import WargameModel
    from wargame_rl.wargame.envs.wargame_objective import WargameObjective


def update_distances_to_objectives(
    wargame_models: list[WargameModel],
    objectives: list[WargameObjective],
    distance_cache: DistanceCache | None = None,
) -> None:
    """Update each model's distances_to_objectives from current locations. Mutates models."""
    if distance_cache is not None:
        deltas = distance_cache.model_obj_deltas.astype(int)
        for i, model in enumerate(wargame_models):
            model.distances_to_objectives = deltas[i]
        return

    for model in wargame_models:
        model.distances_to_objectives = np.array(
            [model.location - obj.location for obj in objectives],
            dtype=int,
        )


def build_observation(
    current_turn: int,
    wargame_models: list[WargameModel],
    objectives: list[WargameObjective],
    max_groups: int,
    board_width: int,
    board_height: int,
) -> WargameEnvObservation:
    """Build the observation dict from current state."""
    wargame_obs = [
        WargameModelObservation(
            location=model.location,
            distances_to_objectives=model.distances_to_objectives,
            group_id=model.group_id,
            max_groups=max_groups,
        )
        for model in wargame_models
    ]
    objectives_obs = [
        WargameEnvObjectiveObservation(location=obj.location) for obj in objectives
    ]
    return WargameEnvObservation(
        current_turn=current_turn,
        wargame_models=wargame_obs,
        objectives=objectives_obs,
        board_width=board_width,
        board_height=board_height,
    )


def build_info(
    current_turn: int,
    wargame_models: list[WargameModel],
    objectives: list[WargameObjective],
    deployment_zone: tuple[int, int, int, int],
    max_groups: int,
) -> WargameEnvInfo:
    """Build the info dict from current state."""
    wargame_obs = [
        WargameModelObservation(
            location=model.location,
            distances_to_objectives=model.distances_to_objectives,
            group_id=model.group_id,
            max_groups=max_groups,
        )
        for model in wargame_models
    ]
    objectives_obs = [
        WargameEnvObjectiveObservation(location=obj.location) for obj in objectives
    ]
    return WargameEnvInfo(
        current_turn=current_turn,
        wargame_models=wargame_obs,
        objectives=objectives_obs,
        deployment_zone=deployment_zone,
    )
