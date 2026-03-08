"""Build observations and info from current env state.

Extracted so observation shape or content can be varied without touching step/reset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.types import (
    EnvScoreState,
    WargameEnvInfo,
    WargameEnvObjectiveObservation,
    WargameEnvObservation,
    WargameModelObservation,
)

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.wargame_model import WargameModel

from wargame_rl.wargame.envs.env_components.distance_cache import (
    DistanceCache,
    compute_levels_of_control,
)
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


def _models_to_obs(
    models: list[WargameModel], max_groups: int
) -> list[WargameModelObservation]:
    return [
        WargameModelObservation(
            location=m.location,
            distances_to_objectives=m.distances_to_objectives,
            group_id=m.group_id,
            max_groups=max_groups,
        )
        for m in models
    ]


def _closest_opponent_distances(
    objectives: list[WargameObjective],
    opponent_models: list[WargameModel],
    sentinel: float,
) -> np.ndarray:
    """Per-objective min distance from any opponent. Sentinel when no opponents."""
    if not opponent_models:
        return np.full(len(objectives), sentinel, dtype=np.float64)
    opp_locs = np.array([m.location for m in opponent_models], dtype=np.float64)
    obj_locs = np.array([o.location for o in objectives], dtype=np.float64)
    # (n_opponents, n_objectives, 2) deltas
    deltas = opp_locs[:, np.newaxis, :] - obj_locs[np.newaxis, :, :]
    norms = np.linalg.norm(deltas, axis=2)
    return norms.min(axis=0)


def build_observation(
    current_turn: int,
    wargame_models: list[WargameModel],
    objectives: list[WargameObjective],
    max_groups: int,
    board_width: int,
    board_height: int,
    opponent_models: list[WargameModel] | None = None,
    action_mask: np.ndarray | None = None,
    battle_round: int = 1,
    battle_phase_index: int = 0,
    n_rounds: int = 5,
    control_range: float | None = None,
    env_score_state: EnvScoreState | None = None,
    distance_cache: DistanceCache | None = None,
) -> WargameEnvObservation:
    """Build the observation from current state.

    When control_range is set, each objective gets player_level_of_control and
    opponent_level_of_control from compute_levels_of_control. Otherwise both are 0.
    When distance_cache is set, each objective gets closest_player_distance and
    closest_opponent_distance for targeting; otherwise 0.0 and sentinel respectively.
    """
    if env_score_state is None:
        env_score_state = EnvScoreState()
    sentinel = float(WargameObjective.MAX_DISTANCE_FOR_SPACE)
    closest_opp = _closest_opponent_distances(
        objectives, opponent_models or [], sentinel
    )
    if distance_cache is not None and distance_cache.model_obj_norms.shape[0] > 0:
        closest_player = distance_cache.model_obj_norms.min(axis=0)
    else:
        closest_player = np.zeros(len(objectives), dtype=np.float64)

    if control_range is not None and control_range > 0:
        player_loc, opponent_loc = compute_levels_of_control(
            wargame_models, opponent_models or [], objectives, control_range
        )
        objectives_obs = [
            WargameEnvObjectiveObservation(
                location=obj.location,
                player_level_of_control=float(player_loc[i]),
                opponent_level_of_control=float(opponent_loc[i]),
                radius_size=obj.radius_size,
                closest_player_distance=float(closest_player[i]),
                closest_opponent_distance=float(closest_opp[i]),
            )
            for i, obj in enumerate(objectives)
        ]
    else:
        objectives_obs = [
            WargameEnvObjectiveObservation(
                location=obj.location,
                radius_size=obj.radius_size,
                closest_player_distance=float(closest_player[i]),
                closest_opponent_distance=float(closest_opp[i]),
            )
            for i, obj in enumerate(objectives)
        ]
    return WargameEnvObservation(
        current_turn=current_turn,
        wargame_models=_models_to_obs(wargame_models, max_groups),
        objectives=objectives_obs,
        board_width=board_width,
        board_height=board_height,
        opponent_models=_models_to_obs(opponent_models or [], max_groups),
        action_mask=action_mask,
        battle_round=battle_round,
        battle_phase_index=battle_phase_index,
        n_rounds=n_rounds,
        env_score_state=env_score_state,
    )


def build_info(
    current_turn: int,
    wargame_models: list[WargameModel],
    objectives: list[WargameObjective],
    deployment_zone: tuple[int, int, int, int],
    opponent_deployment_zone: tuple[int, int, int, int],
    max_groups: int,
    opponent_models: list[WargameModel] | None = None,
    player_vp: int = 0,
    opponent_vp: int = 0,
) -> WargameEnvInfo:
    """Build the info dict from current state."""
    objectives_obs = [
        WargameEnvObjectiveObservation(
            location=obj.location,
            radius_size=obj.radius_size,
            closest_player_distance=0.0,
            closest_opponent_distance=0.0,
        )
        for obj in objectives
    ]
    return WargameEnvInfo(
        current_turn=current_turn,
        wargame_models=_models_to_obs(wargame_models, max_groups),
        objectives=objectives_obs,
        opponent_models=_models_to_obs(opponent_models or [], max_groups),
        deployment_zone=deployment_zone,
        opponent_deployment_zone=opponent_deployment_zone,
        player_vp=player_vp,
        opponent_vp=opponent_vp,
    )
