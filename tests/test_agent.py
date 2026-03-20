import numpy as np
import pytest

from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.dqn.agent import Agent
from wargame_rl.wargame.model.dqn.experience_replay import ReplayBuffer
from wargame_rl.wargame.model.net import RL_Network


@pytest.fixture
def agent(env: WargameEnv, replay_buffer: ReplayBuffer) -> Agent:
    return Agent(env, replay_buffer)


def test_agent(agent: Agent, policy_net: RL_Network) -> None:
    wargame_config = WargameEnvConfig()
    n_wargame_models = wargame_config.number_of_wargame_models
    agent.reset()
    observation = agent.observation
    assert observation is not None
    random_action = agent.get_action(policy_net, observation, 1)
    policy_action = agent.get_action(policy_net, observation, 0)

    assert isinstance(policy_action, WargameEnvAction)
    assert isinstance(random_action, WargameEnvAction)
    assert np.array(random_action.actions).shape == (n_wargame_models,)
    assert np.array(policy_action.actions).shape == (n_wargame_models,)

    reward, done, _exp = agent.play_step(policy_net, 0)
    assert isinstance(reward, float)
    assert isinstance(done, bool)


def test_run_episode(agent: Agent, policy_net: RL_Network) -> None:
    """run_episode completes and returns (total_reward, steps)."""
    total_reward, steps = agent.run_episode(policy_net, epsilon=0.5, save_steps=False)
    assert isinstance(total_reward, float)
    assert isinstance(steps, int)
    assert steps >= 1
