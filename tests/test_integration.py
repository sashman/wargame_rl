"""Lean integration tests covering gaps identified in the test-suite audit."""

import itertools

from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.types.config import ModelConfig, ObjectiveConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.dataset import RLDataset
from wargame_rl.wargame.model.dqn.experience_replay import ReplayBuffer
from wargame_rl.wargame.types import Experience

# ---------------------------------------------------------------------------
# Legacy Reward class (default code path, no reward_phases)
# ---------------------------------------------------------------------------


def test_legacy_reward_produces_nonzero_rewards() -> None:
    """Default env (no reward_phases) uses the Reward class; verify it returns floats."""
    env = WargameEnv(
        config=WargameEnvConfig(
            render_mode=None,
            board_width=20,
            board_height=20,
            number_of_wargame_models=2,
            number_of_objectives=1,
            objective_radius_size=2,
        )
    )
    env.reset(seed=42)
    rewards = []
    for _ in range(10):
        action = WargameEnvAction(actions=list(env.action_space.sample()))
        _, reward, terminated, _, _ = env.step(action)
        assert isinstance(reward, float)
        rewards.append(reward)
        if terminated:
            break
    assert any(r != 0.0 for r in rewards)


# ---------------------------------------------------------------------------
# Termination: all models at objective
# ---------------------------------------------------------------------------


def test_termination_when_all_at_objective() -> None:
    """Place models on the objective; a single step should terminate."""
    cfg = WargameEnvConfig(
        render_mode=None,
        board_width=20,
        board_height=20,
        number_of_wargame_models=2,
        number_of_objectives=1,
        objective_radius_size=2,
        models=[
            ModelConfig(x=10, y=10, group_id=0),
            ModelConfig(x=10, y=10, group_id=0),
        ],
        objectives=[ObjectiveConfig(x=10, y=10)],
    )
    env = WargameEnv(config=cfg)
    env.reset(seed=42)
    action = WargameEnvAction(actions=[0, 0])  # stay
    _, _, terminated, _, _ = env.step(action)
    assert terminated is True


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------


def test_replay_buffer_capacity_overflow(experiences: list[Experience]) -> None:
    """Buffer evicts oldest items when capacity is exceeded."""
    capacity = 3
    buffer = ReplayBuffer(capacity)
    for exp in experiences:
        buffer.append(exp)
    assert len(buffer) == capacity


def test_replay_buffer_sample_batch_larger_than_buffer(
    experiences: list[Experience],
) -> None:
    """sample_batch returns all items when batch_size > buffer length."""
    buffer = ReplayBuffer(100)
    for exp in experiences[:5]:
        buffer.append(exp)
    batch = buffer.sample_batch(999)
    assert len(batch) == 5


# ---------------------------------------------------------------------------
# RLDataset
# ---------------------------------------------------------------------------


def test_rl_dataset_empty_buffer_yields_nothing() -> None:
    """An empty ReplayBuffer means the dataset iterator yields nothing."""
    buffer = ReplayBuffer(100)
    dataset = RLDataset(buffer, sample_size=10)
    items = list(itertools.islice(dataset, 1))
    assert items == []


# ---------------------------------------------------------------------------
# Config backward compatibility (size_to_width_height validator)
# ---------------------------------------------------------------------------


class TestConfigBackwardCompat:
    def test_size_key_sets_width_and_height(self) -> None:
        cfg = WargameEnvConfig.model_validate({"size": 30, "objective_radius_size": 1})
        assert cfg.board_width == 30
        assert cfg.board_height == 30

    def test_width_height_keys(self) -> None:
        cfg = WargameEnvConfig.model_validate(
            {"width": 25, "height": 40, "objective_radius_size": 1}
        )
        assert cfg.board_width == 25
        assert cfg.board_height == 40
