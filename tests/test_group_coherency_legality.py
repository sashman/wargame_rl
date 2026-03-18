from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs.env_components.actions import ActionHandler
from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.types.config import ModelConfig, ObjectiveConfig
from wargame_rl.wargame.envs.wargame import WargameEnv


def _base_config(*, enforce_flag: bool) -> WargameEnvConfig:
    return WargameEnvConfig(
        render_mode=None,
        board_width=10,
        board_height=10,
        number_of_wargame_models=2,
        number_of_objectives=1,
        objective_radius_size=1,
        number_of_battle_rounds=5,
        # Keep movement discretisation stable for the action we pick below.
        n_movement_angles=16,
        n_speed_bins=6,
        max_move_speed=6.0,
        group_max_distance=1.0,
        enforce_group_coherency_legality=enforce_flag,
        models=[
            ModelConfig(x=0, y=0, group_id=0),
            ModelConfig(x=0, y=1, group_id=0),
        ],
        objectives=[ObjectiveConfig(x=9, y=9)],
        # Default skip_phases makes each env.step() cover the movement phase.
    )


def _north_move_action(*, handler: ActionHandler, speed_idx: int) -> int:
    """Return an action moving "north" for the given speed bin index."""
    # With the default 16 movement angles, index 4 corresponds to pi/2 (north).
    angle_idx = 4
    return handler.encode_action(angle_idx=angle_idx, speed_idx=speed_idx)


def test_illegal_move_reverts_to_previous_location_when_enabled() -> None:
    cfg = _base_config(enforce_flag=True)
    env = WargameEnv(config=cfg)
    env.reset(seed=0)

    assert np.array_equal(env.wargame_models[0].location, [0, 0])
    assert np.array_equal(env.wargame_models[1].location, [0, 1])

    handler = ActionHandler(cfg)
    move_far_north = _north_move_action(handler=handler, speed_idx=4)  # dy=5

    env.step(WargameEnvAction(actions=[move_far_north, 0]))  # model 1 stays

    # dy=5 breaks coherency with group_max_distance=1.0, so the move is reverted.
    assert np.array_equal(env.wargame_models[0].location, [0, 0])
    assert np.array_equal(env.wargame_models[1].location, [0, 1])


def test_illegal_move_does_not_revert_when_disabled() -> None:
    cfg = _base_config(enforce_flag=False)
    env = WargameEnv(config=cfg)
    env.reset(seed=0)

    handler = ActionHandler(cfg)
    move_far_north = _north_move_action(handler=handler, speed_idx=4)  # dy=5

    env.step(WargameEnvAction(actions=[move_far_north, 0]))  # model 1 stays

    # With the legality check disabled, the move should apply normally.
    assert np.array_equal(env.wargame_models[0].location, [0, 5])
    assert np.array_equal(env.wargame_models[1].location, [0, 1])


def test_legal_move_is_allowed_when_enabled() -> None:
    cfg = _base_config(enforce_flag=True)
    env = WargameEnv(config=cfg)
    env.reset(seed=0)

    handler = ActionHandler(cfg)
    move_north_by_1 = _north_move_action(handler=handler, speed_idx=0)  # dy=1

    env.step(WargameEnvAction(actions=[move_north_by_1, 0]))  # model 1 stays

    # dy=1 results in overlapping positions, which is coherent (distance 0).
    assert np.array_equal(env.wargame_models[0].location, [0, 1])
    assert np.array_equal(env.wargame_models[1].location, [0, 1])
