"""Basic tests for WargameEnv reset and step."""

import pytest

from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv


@pytest.fixture
def env_config() -> WargameEnvConfig:
    return WargameEnvConfig(
        render_mode=None,
        board_width=10,
        board_height=10,
        number_of_wargame_models=2,
        number_of_objectives=2,
        objective_radius_size=1,
    )


@pytest.fixture
def env(env_config: WargameEnvConfig) -> WargameEnv:
    return WargameEnv(config=env_config)


# --- Reset tests ---


def test_reset_returns_observation_and_info(env: WargameEnv) -> None:
    """reset() returns (observation, info) where info is a dict."""
    observation, info = env.reset(seed=42)
    assert observation is not None
    assert isinstance(info, dict)


def test_reset_observation_has_expected_structure(env: WargameEnv) -> None:
    """Observation has current_turn, wargame_models, and objectives."""
    observation, _ = env.reset(seed=42)
    assert hasattr(observation, "current_turn")
    assert hasattr(observation, "wargame_models")
    assert hasattr(observation, "objectives")
    assert observation.current_turn == 0
    assert len(observation.wargame_models) == env.config.number_of_wargame_models
    assert len(observation.objectives) == env.config.number_of_objectives


def test_reset_with_seed_is_reproducible(env: WargameEnv) -> None:
    """Same seed produces same initial observation (model/objective positions)."""
    obs1, _ = env.reset(seed=123)
    obs2, _ = env.reset(seed=123)
    assert obs1.current_turn == obs2.current_turn == 0
    for m1, m2 in zip(obs1.wargame_models, obs2.wargame_models):
        assert (m1.location == m2.location).all()
    for o1, o2 in zip(obs1.objectives, obs2.objectives):
        assert (o1.location == o2.location).all()


def test_reset_info_contains_expected_keys(env: WargameEnv) -> None:
    """info dict contains current_turn, wargame_models, objectives, deployment zones."""
    _, info = env.reset(seed=42)
    assert "current_turn" in info
    assert "wargame_models" in info
    assert "objectives" in info
    assert "deployment_zone" in info
    assert "opponent_deployment_zone" in info


def test_reset_sets_internal_state(env: WargameEnv) -> None:
    """After reset, env current_turn is 0 and last_reward is None."""
    env.reset(seed=42)
    assert env.current_turn == 0
    assert env.last_reward is None


# --- Step tests ---


def test_step_returns_five_tuple(env: WargameEnv) -> None:
    """step() returns (observation, reward, terminated, truncated, info)."""
    env.reset(seed=42)
    action = WargameEnvAction(actions=[0, 0])  # right, right
    result = env.step(action)
    assert len(result) == 5
    observation, reward, terminated, truncated, info = result
    assert observation is not None
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_step_observation_structure(env: WargameEnv) -> None:
    """Step observation has same structure as reset observation."""
    env.reset(seed=42)
    action = WargameEnvAction(actions=[0, 1])
    observation, _, _, _, _ = env.step(action)
    assert hasattr(observation, "current_turn")
    assert hasattr(observation, "wargame_models")
    assert hasattr(observation, "objectives")
    assert len(observation.wargame_models) == env.config.number_of_wargame_models
    assert len(observation.objectives) == env.config.number_of_objectives


def test_step_increments_turn(env: WargameEnv) -> None:
    """After one step, env.current_turn is 1."""
    env.reset(seed=42)
    assert env.current_turn == 0
    env.step(WargameEnvAction(actions=[0, 0]))
    assert env.current_turn == 1


def test_step_updates_last_reward(env: WargameEnv) -> None:
    """After step, last_reward is set to the returned reward."""
    env.reset(seed=42)
    assert env.last_reward is None
    _, reward, _, _, _ = env.step(WargameEnvAction(actions=[0, 0]))
    assert env.last_reward == reward


def test_step_invalid_action_raises(env: WargameEnv) -> None:
    """Action out of bounds for a model raises ValueError."""
    env.reset(seed=42)
    # Assume Discrete(5) per model; 99 is invalid
    invalid_action = WargameEnvAction(actions=[99, 0])
    with pytest.raises(ValueError, match="out of bounds"):
        env.step(invalid_action)


def test_step_respects_action_space(env: WargameEnv) -> None:
    """Sampling from action_space and stepping works for several steps."""
    env.reset(seed=42)
    for _ in range(5):
        action = WargameEnvAction(actions=list(env.action_space.sample()))
        observation, reward, terminated, truncated, info = env.step(action)
        assert observation is not None
        assert isinstance(reward, float)
        if terminated or truncated:
            break


def test_multiple_steps(env: WargameEnv) -> None:
    """Run many steps: turn increments, valid returns each step, episode eventually ends."""
    observation, info = env.reset(seed=42)
    assert env.current_turn == 0

    max_steps = 100
    step_count = 0
    for _ in range(max_steps):
        action = WargameEnvAction(actions=list(env.action_space.sample()))
        observation, reward, terminated, truncated, info = env.step(action)

        step_count += 1
        assert observation is not None
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert env.current_turn == step_count
        assert len(observation.wargame_models) == env.config.number_of_wargame_models
        assert len(observation.objectives) == env.config.number_of_objectives

        if terminated or truncated:
            break

    assert step_count >= 1
    assert env.current_turn == step_count
