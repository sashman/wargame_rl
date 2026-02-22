import gymnasium as gym
import numpy as np
import pytest

from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.model.dqn.agent import Agent
from wargame_rl.wargame.model.dqn.experience_replay import ReplayBuffer
from wargame_rl.wargame.model.net import MLPNetwork


@pytest.fixture
def agent(env: gym.Env, replay_buffer: ReplayBuffer) -> Agent:
    return Agent(env, replay_buffer)


def test_agent(agent: Agent, policy_net: MLPNetwork) -> None:
    wargame_config = WargameEnvConfig()
    n_wargame_models = wargame_config.number_of_wargame_models
    agent.reset()
    random_action = agent.get_action(policy_net, 1)
    policy_action = agent.get_action(policy_net, 0)

    assert isinstance(policy_action, WargameEnvAction)
    assert isinstance(random_action, WargameEnvAction)
    assert np.array(random_action.actions).shape == (n_wargame_models,)
    assert np.array(policy_action.actions).shape == (n_wargame_models,)

    reward, done = agent.play_step(policy_net, 0)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
