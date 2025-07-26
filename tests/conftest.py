from functools import lru_cache

import pytest
import torch

from wargame_rl.wargame.model.dqn.dqn import DQN
from wargame_rl.wargame.model.dqn.experience_replay import ReplayBuffer
from wargame_rl.wargame.model.dqn.factory import create_environment
from wargame_rl.wargame.types import ExperienceV1


@pytest.fixture
def n_steps() -> int:
    return 10


@pytest.fixture
@lru_cache(maxsize=1)
def env():
    return create_environment(render_mode=None)


@pytest.fixture
@lru_cache(maxsize=1)
def experiences(env, n_steps: int) -> list[ExperienceV1]:
    previous_state, _ = env.reset()
    output = []

    for _ in range(n_steps):
        action = env.action_space.sample()
        state, reward, terminated, _, _ = env.step(action)
        output.append(ExperienceV1(previous_state, action, reward, terminated, state))
        previous_state = state

    return output


@pytest.fixture
def replay_buffer(n_steps: int, experiences: list[ExperienceV1]) -> ReplayBuffer:
    buffer = ReplayBuffer(n_steps)
    for experience in experiences:
        buffer.append(experience)
    return buffer


@pytest.fixture
@lru_cache(maxsize=1)
def dqn_net(env) -> torch.nn.Module:
    dqn_net = DQN.from_env(env)
    return dqn_net
