import pytest
import gymnasium as gym
from functools import lru_cache
from wargame_rl.wargame.types import Experience
from wargame_rl.wargame.model.dqn.experience_replay import ReplayBuffer
from wargame_rl.wargame.model.dqn.dqn import DQN
import torch
from gymnasium.spaces.utils import flatten_space


@pytest.fixture
def n_steps() -> int:
    return 10


@pytest.fixture
@lru_cache(maxsize=1)
def env():
    return gym.make("gymnasium_env/Wargame-v0", render_mode=None)


@pytest.fixture
@lru_cache(maxsize=1)
def experiences(env, n_steps: int) -> list[Experience]:
    previous_state, _ = env.reset()
    output = []

    for _ in range(n_steps):
        action = env.action_space.sample()
        state, reward, terminated, _, _ = env.step(action)
        output.append(Experience(previous_state, action, reward, terminated, state))
        previous_state = state

    return output


@pytest.fixture
def replay_buffer(n_steps: int, experiences: list[Experience]) -> ReplayBuffer:
    buffer = ReplayBuffer(n_steps)
    for experience in experiences:
        buffer.append(experience)
    return buffer


@pytest.fixture
@lru_cache(maxsize=1)
def dqn_net(env) -> torch.nn.Module:
    # Build the network
    obs_size = flatten_space(env.observation_space).shape[0]
    n_actions = env.action_space.n
    net = DQN(obs_size, n_actions)
    return net
