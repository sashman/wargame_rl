"""Build observations and info from battle state (BattleView).

Extracted so observation shape or content can be varied without touching step/reset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.domain.battle_view import BattleView
from wargame_rl.wargame.envs.domain.entities import alive_mask_for
from wargame_rl.wargame.envs.env_components.actions import ActionRegistry
from wargame_rl.wargame.envs.types import (
    WargameEnvInfo,
    WargameEnvObjectiveObservation,
    WargameEnvObservation,
    WargameModelObservation,
)
from wargame_rl.wargame.envs.types.game_timing import BattlePhase

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


def _models_to_obs(
    models: list[WargameModel], max_groups: int
) -> list[WargameModelObservation]:
    return [
        WargameModelObservation(
            location=m.location,
            distances_to_objectives=m.distances_to_objectives,
            group_id=m.group_id,
            max_groups=max_groups,
            alive=1.0 if m.is_alive else 0.0,
            current_wounds=int(m.stats["current_wounds"]),
            max_wounds=int(m.stats["max_wounds"]),
        )
        for m in models
    ]


def build_observation(
    view: BattleView,
    distance_cache: DistanceCache | None = None,
    action_registry: ActionRegistry | None = None,
) -> WargameEnvObservation:
    """Build the Gym observation from battle state (BattleView)."""
    if distance_cache is not None:
        update_distances_to_objectives(
            view.player_models, view.objectives, distance_cache
        )
    if view.opponent_models:
        update_distances_to_objectives(view.opponent_models, view.objectives, None)

    action_mask: np.ndarray | None = None
    if action_registry is not None:
        phase = view.game_clock_state.phase or BattlePhase.movement
        player_alive = alive_mask_for(view.player_models)
        action_mask = action_registry.get_model_action_masks(
            phase, len(view.player_models), alive_mask=player_alive
        )

    clock = view.game_clock_state
    phase = clock.phase or BattlePhase.movement
    battle_phase_index = list(BattlePhase).index(phase)
    battle_round = clock.battle_round if clock.battle_round is not None else 1
    max_groups = view.config.max_groups
    objectives_obs = [
        WargameEnvObjectiveObservation(location=obj.location) for obj in view.objectives
    ]
    return WargameEnvObservation(
        current_turn=view.current_turn,
        wargame_models=_models_to_obs(view.player_models, max_groups),
        objectives=objectives_obs,
        board_width=view.board_width,
        board_height=view.board_height,
        opponent_models=_models_to_obs(view.opponent_models, max_groups),
        action_mask=action_mask,
        battle_round=battle_round,
        battle_phase_index=battle_phase_index,
        n_rounds=view.n_rounds,
        player_vp=view.player_vp,
        opponent_vp=view.opponent_vp,
        player_vp_delta=view.player_vp_delta,
    )


def build_info(view: BattleView) -> WargameEnvInfo:
    """Build the Gym info dict from battle state (BattleView)."""
    dz = view.deployment_zone
    odz = view.opponent_deployment_zone
    deployment_zone = (int(dz[0]), int(dz[1]), int(dz[2]), int(dz[3]))
    opponent_deployment_zone = (int(odz[0]), int(odz[1]), int(odz[2]), int(odz[3]))
    max_groups = view.config.max_groups
    objectives_obs = [
        WargameEnvObjectiveObservation(location=obj.location) for obj in view.objectives
    ]
    return WargameEnvInfo(
        current_turn=view.current_turn,
        wargame_models=_models_to_obs(view.player_models, max_groups),
        objectives=objectives_obs,
        opponent_models=_models_to_obs(view.opponent_models, max_groups),
        deployment_zone=deployment_zone,
        opponent_deployment_zone=opponent_deployment_zone,
        player_vp=view.player_vp,
        opponent_vp=view.opponent_vp,
        player_vp_delta=view.player_vp_delta,
        opponent_vp_delta=view.opponent_vp_delta,
    )
