import pytest
import gymnasium as gym
from wargame_rl.wargame.model.dqn.agent import Agent
from wargame_rl.wargame.model.dqn.experience_replay import ReplayBuffer
from wargame_rl.wargame.model.dqn.dqn import DQN


@pytest.fixture
def agent(env: gym.Env, replay_buffer: ReplayBuffer) -> Agent:
    return Agent(env, replay_buffer)


def test_agent(agent, dqn_net: DQN):
    agent.reset()
    random_action = agent.get_action(dqn_net, 1)
    dqn_action = agent.get_action(dqn_net, 0)
    assert type(random_action) == int
    assert type(dqn_action) == int
