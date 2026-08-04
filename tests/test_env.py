"""Basic tests for WargameEnv reset and step."""

import pytest

from wargame_rl.wargame.envs.reward.phase import (
    RewardCalculatorConfig,
    RewardPhaseConfig,
    SuccessCriteriaConfig,
)
from wargame_rl.wargame.envs.types import (
    TerrainPieceConfig,
    WargameEnvAction,
    WargameEnvConfig,
)
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
        number_of_battle_rounds=100,
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
    """Observation has current_turn, wargame_models, objectives, and VP."""
    observation, _ = env.reset(seed=42)
    assert hasattr(observation, "current_turn")
    assert hasattr(observation, "wargame_models")
    assert hasattr(observation, "objectives")
    assert hasattr(observation, "player_vp")
    assert hasattr(observation, "opponent_vp")
    assert observation.current_turn == 0
    assert observation.player_vp == 0
    assert observation.opponent_vp == 0
    assert observation.size_game_observation == 6
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
    """info dict contains current_turn, wargame_models, objectives, deployment zones, VP."""
    _, info = env.reset(seed=42)
    assert "current_turn" in info
    assert "wargame_models" in info
    assert "objectives" in info
    assert "deployment_zone" in info
    assert "opponent_deployment_zone" in info
    assert "player_vp" in info
    assert "opponent_vp" in info
    assert info["player_vp"] == 0
    assert info["opponent_vp"] == 0


def test_reset_sets_internal_state(env: WargameEnv) -> None:
    """After reset, env current_turn is 0 and last_reward is None."""
    env.reset(seed=42)
    assert env.current_turn == 0
    assert env.last_reward is None


def test_reset_clears_previous_closest_objective_distance(env: WargameEnv) -> None:
    """reset() clears per-episode distance memory used by shaped rewards."""
    env.reset(seed=42)
    env.step(WargameEnvAction(actions=[0, 0]))
    assert any(
        m.previous_closest_objective_distance is not None for m in env.wargame_models
    )

    env.reset(seed=42)
    assert all(
        m.previous_closest_objective_distance is None for m in env.wargame_models
    )


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


def test_step_adds_terminal_success_bonus_when_all_models_at_objective() -> None:
    config = WargameEnvConfig(
        render_mode=None,
        board_width=10,
        board_height=10,
        number_of_wargame_models=1,
        number_of_objectives=1,
        objective_radius_size=1,
        models=[{"x": 4, "y": 4, "group_id": 0}],  # type: ignore[list-item]
        objectives=[{"x": 4, "y": 4}],  # type: ignore[list-item]
        reward_phases=[
            RewardPhaseConfig(
                name="reach",
                reward_calculators=[
                    RewardCalculatorConfig(type="closest_objective", weight=1.0),
                ],
                success_criteria=SuccessCriteriaConfig(type="all_at_objectives"),
                terminal_success_bonus=12.5,
            ),
        ],
    )
    env = WargameEnv(config=config)
    env.reset(seed=42)
    _, reward, terminated, _, _ = env.step(WargameEnvAction(actions=[0]))
    assert terminated is True
    assert reward >= 12.5


def _fraction_termination_config(
    *, min_fraction: float, terminate_on_success: bool
) -> WargameEnvConfig:
    """Two models, one on an objective and one far away -> fraction is exactly 0.5."""
    return WargameEnvConfig(
        render_mode=None,
        board_width=10,
        board_height=10,
        number_of_wargame_models=2,
        number_of_objectives=1,
        objective_radius_size=1,
        models=[
            {"x": 4, "y": 4, "group_id": 0},  # type: ignore[list-item]
            {"x": 9, "y": 9, "group_id": 0},  # type: ignore[list-item]
        ],
        objectives=[{"x": 4, "y": 4}],  # type: ignore[list-item]
        reward_phases=[
            RewardPhaseConfig(
                name="reach",
                reward_calculators=[
                    RewardCalculatorConfig(type="closest_objective", weight=1.0),
                ],
                success_criteria=SuccessCriteriaConfig(
                    type="fraction_at_objectives",
                    params={"min_fraction": min_fraction},
                ),
                terminate_on_success=terminate_on_success,
            ),
        ],
    )


def test_terminates_on_success_of_configured_fraction_criteria() -> None:
    """Termination consults the phase's criteria, not a hardcoded all-at-objectives.

    With 1 of 2 models on the objective the fraction is 0.5, which satisfies a
    0.5 criteria but never satisfies all-at-objectives -- so this episode could
    not end early before termination honoured the configured criteria.
    """
    env = WargameEnv(
        config=_fraction_termination_config(min_fraction=0.5, terminate_on_success=True)
    )
    env.reset(seed=42)

    _, _, terminated, _, _ = env.step(WargameEnvAction(actions=[0, 0]))

    assert terminated is True


def test_does_not_terminate_when_fraction_criteria_unmet() -> None:
    """A fraction the state does not reach must not end the episode."""
    env = WargameEnv(
        config=_fraction_termination_config(min_fraction=1.0, terminate_on_success=True)
    )
    env.reset(seed=42)

    _, _, terminated, _, _ = env.step(WargameEnvAction(actions=[0, 0]))

    assert terminated is False


def test_terminate_on_success_false_keeps_episode_running_on_success() -> None:
    """`terminate_on_success: false` must suppress success termination entirely."""
    env = WargameEnv(
        config=_fraction_termination_config(
            min_fraction=0.5, terminate_on_success=False
        )
    )
    env.reset(seed=42)

    _, _, terminated, _, _ = env.step(WargameEnvAction(actions=[0, 0]))

    assert terminated is False


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


# --- Terrain movement tests ---


def test_terrain_movement_through_footprint() -> None:
    """Model can move into/through/occupy footprint cells (TERR-05)."""
    config = WargameEnvConfig(
        render_mode=None,
        board_width=20,
        board_height=20,
        number_of_wargame_models=1,
        number_of_objectives=1,
        objective_radius_size=1,
        terrain=[TerrainPieceConfig(footprint=(8, 8, 12, 12))],
        number_of_battle_rounds=100,
    )
    env = WargameEnv(config=config)
    env.reset(seed=42)
    for _ in range(20):
        action = WargameEnvAction(actions=list(env.action_space.sample()))
        obs, _, terminated, _, _ = env.step(action)
        if terminated:
            break
    # Verify env didn't crash — terrain does not block movement
    assert env.current_turn >= 1
