import gymnasium as gym
import numpy as np
import pytest

from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.model.dqn.agent import Agent
from wargame_rl.wargame.model.dqn.dqn import DQN
from wargame_rl.wargame.model.dqn.experience_replay import ReplayBuffer


@pytest.fixture
def agent(env: gym.Env, replay_buffer: ReplayBuffer) -> Agent:
    return Agent(env, replay_buffer)


def test_agent(agent: Agent, dqn_net: DQN) -> None:
    wargame_config = WargameEnvConfig()
    n_wargame_models = wargame_config.number_of_wargame_models
    agent.reset()
    random_action = agent.get_action(dqn_net, 1)
    dqn_action = agent.get_action(dqn_net, 0)

    assert isinstance(dqn_action, WargameEnvAction)
    assert isinstance(random_action, WargameEnvAction)
    assert np.array(random_action.actions).shape == (n_wargame_models,)
    assert np.array(dqn_action.actions).shape == (n_wargame_models,)

    reward, done = agent.play_step(dqn_net, 0)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
