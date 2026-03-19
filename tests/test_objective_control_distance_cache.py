from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs.env_components.distance_cache import (
    compute_distances,
    objective_ownership_from_norms_offset,
)
from wargame_rl.wargame.envs.types import OpponentPolicyConfig, WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv


def test_objective_control_includes_radius_boundary() -> None:
    """Objective is controlled when distance == radius_size."""
    cfg = WargameEnvConfig(
        render_mode=None,
        board_width=20,
        board_height=20,
        number_of_wargame_models=1,
        number_of_opponent_models=1,
        number_of_objectives=1,
        objective_radius_size=2,
        opponent_policy=OpponentPolicyConfig(type="random"),
        number_of_battle_rounds=5,
    )
    env = WargameEnv(config=cfg)
    env.reset(seed=0)

    env.objectives[0].location = np.array([10, 10], dtype=np.int32)
    env.wargame_models[0].location = np.array([10, 12], dtype=np.int32)  # dist 2
    env.opponent_models[0].location = np.array([0, 0], dtype=np.int32)

    player_cache = compute_distances(env.wargame_models, env.objectives)
    opponent_cache = compute_distances(env.opponent_models, env.objectives)
    player_controls, opponent_controls = objective_ownership_from_norms_offset(
        player_cache.model_obj_norms_offset,
        opponent_cache.model_obj_norms_offset,
        player_cache.obj_radii,
    )

    assert bool(player_controls[0]) is True
    assert bool(opponent_controls[0]) is False


def test_objective_control_excludes_just_outside_radius() -> None:
    """Objective is not controlled when distance == radius_size + 1."""
    cfg = WargameEnvConfig(
        render_mode=None,
        board_width=20,
        board_height=20,
        number_of_wargame_models=1,
        number_of_opponent_models=1,
        number_of_objectives=1,
        objective_radius_size=2,
        opponent_policy=OpponentPolicyConfig(type="random"),
        number_of_battle_rounds=5,
    )
    env = WargameEnv(config=cfg)
    env.reset(seed=0)

    env.objectives[0].location = np.array([10, 10], dtype=np.int32)
    env.wargame_models[0].location = np.array([10, 13], dtype=np.int32)  # dist 3
    env.opponent_models[0].location = np.array([0, 0], dtype=np.int32)

    player_cache = compute_distances(env.wargame_models, env.objectives)
    opponent_cache = compute_distances(env.opponent_models, env.objectives)
    player_controls, opponent_controls = objective_ownership_from_norms_offset(
        player_cache.model_obj_norms_offset,
        opponent_cache.model_obj_norms_offset,
        player_cache.obj_radii,
    )

    assert bool(player_controls[0]) is False
    assert bool(opponent_controls[0]) is False
