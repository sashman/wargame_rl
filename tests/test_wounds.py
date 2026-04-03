"""Tests for wound tracking, elimination, and termination."""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.envs.domain.entities import WargameModel, alive_mask_for
from wargame_rl.wargame.envs.domain.game_clock import GameClock
from wargame_rl.wargame.envs.domain.termination import is_battle_over
from wargame_rl.wargame.envs.env_components.distance_cache import compute_distances
from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.types.config import (
    ModelConfig,
    ObjectiveConfig,
    OpponentPolicyConfig,
)
from wargame_rl.wargame.envs.wargame import WargameEnv


def _make_model(max_wounds: int, current_wounds: int | None = None) -> WargameModel:
    """Helper: create a WargameModel with wound stats."""
    cw = current_wounds if current_wounds is not None else max_wounds
    return WargameModel(
        location=np.array([0, 0], dtype=np.int32),
        stats={"max_wounds": max_wounds, "current_wounds": cw},
        distances_to_objectives=np.zeros((1, 2), dtype=np.int32),
        group_id=0,
    )


@pytest.mark.parametrize(
    "max_wounds, damage, expected_current",
    [
        (1, 1, 0),
        (2, 1, 1),
        (3, 2, 1),
    ],
)
def test_wound_tracking(max_wounds: int, damage: int, expected_current: int) -> None:
    model = _make_model(max_wounds)
    model.take_damage(damage)
    assert model.stats["current_wounds"] == expected_current


def test_take_damage_clamps_at_zero() -> None:
    model = _make_model(max_wounds=2)
    model.take_damage(10)
    assert model.stats["current_wounds"] == 0


def test_is_alive_true_when_wounded() -> None:
    model = _make_model(max_wounds=2)
    assert model.is_alive is True
    model.take_damage(1)
    assert model.is_alive is True


def test_is_alive_false_when_eliminated() -> None:
    model = _make_model(max_wounds=1)
    model.take_damage(1)
    assert model.is_alive is False


def test_reset_for_episode_restores_wounds() -> None:
    model = _make_model(max_wounds=3)
    model.take_damage(2)
    assert model.stats["current_wounds"] == 1
    model.reset_for_episode()
    assert model.stats["current_wounds"] == 3
    assert model.is_alive is True


def test_is_battle_over_all_eliminated() -> None:
    clock = GameClock(n_rounds=5)
    result = is_battle_over(
        clock=clock,
        current_turn=0,
        max_turns=100,
        max_turns_override=100,
        all_models_at_objectives_flag=False,
        all_eliminated=True,
    )
    assert result is True


def test_is_battle_over_without_elimination() -> None:
    clock = GameClock(n_rounds=5)
    result = is_battle_over(
        clock=clock,
        current_turn=0,
        max_turns=100,
        max_turns_override=100,
        all_models_at_objectives_flag=False,
        all_eliminated=False,
    )
    assert result is False


def test_config_default_max_wounds() -> None:
    assert ModelConfig().max_wounds == 1


def test_config_explicit_max_wounds() -> None:
    assert ModelConfig(max_wounds=3).max_wounds == 3


def test_alive_mask_for() -> None:
    alive = _make_model(max_wounds=2)
    dead = _make_model(max_wounds=1)
    dead.take_damage(1)
    result = alive_mask_for([alive, dead])
    np.testing.assert_array_equal(result, np.array([True, False]))


# ---------------------------------------------------------------------------
# Integration test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wound_env() -> WargameEnv:
    """Env with 2 player models (max_wounds=2), 1 objective, no opponents."""
    config = WargameEnvConfig(
        board_width=20,
        board_height=20,
        number_of_wargame_models=2,
        number_of_objectives=1,
        objective_radius_size=2,
        models=[
            ModelConfig(x=5, y=5, max_wounds=2),
            ModelConfig(x=10, y=10, max_wounds=2),
        ],
        objectives=[ObjectiveConfig(x=10, y=10, radius_size=2)],
        max_turns_override=50,
        number_of_battle_rounds=5,
    )
    return WargameEnv(config=config)


@pytest.fixture
def wound_env_with_opponents() -> WargameEnv:
    """Env with 2 player models and 2 opponent models, all 1 wound."""
    config = WargameEnvConfig(
        board_width=20,
        board_height=20,
        number_of_wargame_models=2,
        number_of_objectives=1,
        objective_radius_size=2,
        models=[
            ModelConfig(x=2, y=2, max_wounds=1),
            ModelConfig(x=3, y=3, max_wounds=1),
        ],
        objectives=[ObjectiveConfig(x=10, y=10, radius_size=2)],
        number_of_opponent_models=2,
        opponent_models=[
            ModelConfig(x=17, y=17, max_wounds=1),
            ModelConfig(x=18, y=18, max_wounds=1),
        ],
        opponent_policy=OpponentPolicyConfig(type="scripted_advance_to_objective"),
        max_turns_override=50,
        number_of_battle_rounds=5,
    )
    return WargameEnv(config=config)


# ---------------------------------------------------------------------------
# Integration tests: eliminated model exclusion
# ---------------------------------------------------------------------------


def test_eliminated_model_does_not_move(wound_env: WargameEnv) -> None:
    """Dead models must not move regardless of the action given."""
    wound_env.reset(seed=42)
    wound_env.wargame_models[0].take_damage(2)
    frozen_loc = wound_env.wargame_models[0].location.copy()

    move_action = wound_env._action_handler.encode_action(0, 0)
    wound_env.step(WargameEnvAction(actions=[move_action, 0]))

    np.testing.assert_array_equal(wound_env.wargame_models[0].location, frozen_loc)


def test_eliminated_model_not_controlling_objective() -> None:
    """A dead model at an objective should not count for objective control.

    Model 0 is placed at the objective and killed. Model 1 is alive but far
    away. all_models_at_objectives should return False because the alive model
    is not at the objective.
    """
    config = WargameEnvConfig(
        board_width=20,
        board_height=20,
        number_of_wargame_models=2,
        number_of_objectives=1,
        objective_radius_size=2,
        models=[
            ModelConfig(x=10, y=10, max_wounds=2),
            ModelConfig(x=1, y=1, max_wounds=2),
        ],
        objectives=[ObjectiveConfig(x=10, y=10, radius_size=2)],
        max_turns_override=50,
        number_of_battle_rounds=5,
    )
    env = WargameEnv(config=config)
    env.reset(seed=42)

    env.wargame_models[0].take_damage(2)
    alive = alive_mask_for(env.wargame_models)
    cache = compute_distances(env.wargame_models, env.objectives, alive_mask=alive)
    assert cache.all_models_at_objectives(alive_mask=alive) is False


def test_player_elimination_does_not_terminate_by_default(
    wound_env: WargameEnv,
) -> None:
    """Without terminate_on_player_elimination, wiping the player does not end the game."""
    wound_env.reset(seed=42)
    for m in wound_env.wargame_models:
        m.take_damage(m.stats["max_wounds"])

    _, _, terminated, _, _ = wound_env.step(WargameEnvAction(actions=[0, 0]))
    assert terminated is False


def test_player_elimination_terminates_when_flag_set() -> None:
    """With terminate_on_player_elimination=True, wiping the player ends the episode."""
    config = WargameEnvConfig(
        board_width=20,
        board_height=20,
        number_of_wargame_models=2,
        number_of_objectives=1,
        objective_radius_size=2,
        models=[
            ModelConfig(x=5, y=5, max_wounds=2),
            ModelConfig(x=10, y=10, max_wounds=2),
        ],
        objectives=[ObjectiveConfig(x=10, y=10, radius_size=2)],
        max_turns_override=50,
        terminate_on_player_elimination=True,
    )
    env = WargameEnv(config=config)
    env.reset(seed=42)
    for m in env.wargame_models:
        m.take_damage(m.stats["max_wounds"])

    _, _, terminated, _, _ = env.step(WargameEnvAction(actions=[0, 0]))
    assert terminated is True


def test_termination_all_opponent_eliminated(
    wound_env_with_opponents: WargameEnv,
) -> None:
    """Episode must terminate when all opponent models are eliminated."""
    wound_env_with_opponents.reset(seed=42)
    for m in wound_env_with_opponents.opponent_models:
        m.take_damage(m.stats["max_wounds"])

    _, _, terminated, _, _ = wound_env_with_opponents.step(
        WargameEnvAction(actions=[0, 0])
    )
    assert terminated is True


def test_no_vacuous_termination_zero_opponents(
    wound_env: WargameEnv,
) -> None:
    """A 0-opponent config must not trigger vacuous all_eliminated termination."""
    wound_env.reset(seed=42)
    _, _, terminated, _, _ = wound_env.step(WargameEnvAction(actions=[0, 0]))
    assert terminated is False


def test_all_alive_models_at_objectives() -> None:
    """Alive models at objective = success, despite dead models existing."""
    config = WargameEnvConfig(
        board_width=20,
        board_height=20,
        number_of_wargame_models=2,
        number_of_objectives=1,
        objective_radius_size=2,
        models=[
            ModelConfig(x=10, y=10, max_wounds=2),
            ModelConfig(x=10, y=10, max_wounds=2),
        ],
        objectives=[ObjectiveConfig(x=10, y=10, radius_size=2)],
        max_turns_override=50,
        number_of_battle_rounds=5,
    )
    env = WargameEnv(config=config)
    env.reset(seed=42)

    env.wargame_models[0].take_damage(2)

    alive = alive_mask_for(env.wargame_models)
    cache = compute_distances(env.wargame_models, env.objectives, alive_mask=alive)
    assert cache.all_models_at_objectives(alive_mask=alive) is True
