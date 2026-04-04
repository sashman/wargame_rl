"""Helpers to build an observation from the opponent's perspective."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.env_components.observation_builder import (
    update_distances_to_objectives,
)
from wargame_rl.wargame.envs.types import (
    WargameEnvObjectiveObservation,
    WargameEnvObservation,
    WargameModelObservation,
)
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame_model import WargameModel

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.wargame import WargameEnv


def _to_model_observations(
    models: list[WargameModel], max_groups: int
) -> list[WargameModelObservation]:
    return [
        WargameModelObservation(
            location=model.location,
            distances_to_objectives=model.distances_to_objectives,
            group_id=model.group_id,
            max_groups=max_groups,
            alive=1.0 if model.is_alive else 0.0,
            current_wounds=int(model.stats["current_wounds"]),
            max_wounds=int(model.stats["max_wounds"]),
        )
        for model in models
    ]


def build_opponent_observation(
    env: "WargameEnv", action_mask: np.ndarray | None = None
) -> WargameEnvObservation:
    """Build a player-compatible observation where opponent is the acting side."""
    update_distances_to_objectives(env.opponent_models, env.objectives, None)
    update_distances_to_objectives(env.wargame_models, env.objectives, None)

    clock_state = env.game_clock_state
    phase = clock_state.phase or BattlePhase.movement
    battle_phase_index = list(BattlePhase).index(phase)
    battle_round = (
        clock_state.battle_round if clock_state.battle_round is not None else 1
    )
    max_groups = env.config.max_groups

    objectives_obs = [
        WargameEnvObjectiveObservation(location=obj.location) for obj in env.objectives
    ]

    return WargameEnvObservation(
        current_turn=env.current_turn,
        wargame_models=_to_model_observations(env.opponent_models, max_groups),
        objectives=objectives_obs,
        board_width=env.board_width,
        board_height=env.board_height,
        opponent_models=_to_model_observations(env.wargame_models, max_groups),
        action_mask=action_mask,
        battle_round=battle_round,
        battle_phase_index=battle_phase_index,
        n_rounds=env.n_rounds,
        player_vp=env.opponent_vp,
        opponent_vp=env.player_vp,
        player_vp_delta=env.opponent_vp_delta,
    )
