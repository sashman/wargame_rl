import gymnasium as gym
import pytest

from wargame_rl.wargame.model.dqn.agent import Agent
from wargame_rl.wargame.model.dqn.dqn import DQN
from wargame_rl.wargame.model.dqn.experience_replay import ReplayBuffer


@pytest.fixture
def agent(env: gym.Env, replay_buffer: ReplayBuffer) -> Agent:
    return Agent(env, replay_buffer)


def test_agent(agent, dqn_net: DQN):
    agent.reset()
    random_action = agent.get_action(dqn_net, 1)
    dqn_action = agent.get_action(dqn_net, 0)
    assert isinstance(dqn_action, int)
    assert isinstance(random_action, int)

    reward, done = agent.play_step(dqn_net, 0)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
