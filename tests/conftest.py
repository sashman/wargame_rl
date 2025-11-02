from functools import lru_cache

import pytest

from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.dqn.dqn import DQN_MLP
from wargame_rl.wargame.model.dqn.experience_replay import ReplayBuffer
from wargame_rl.wargame.types import Experience


@pytest.fixture
def n_steps() -> int:
    return 10


@pytest.fixture
@lru_cache(maxsize=1)
def env() -> WargameEnv:
    return WargameEnv(config=WargameEnvConfig(render_mode=None))


@pytest.fixture
@lru_cache(maxsize=1)
def experiences(env: WargameEnv, n_steps: int) -> list[Experience]:
    previous_state, _ = env.reset()
    output = []

    for _ in range(n_steps):
        action = WargameEnvAction(actions=env.action_space.sample())
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
def dqn_net(env: WargameEnv) -> DQN_MLP:
    return DQN_MLP.from_env(env=env)
