from typing import Tuple

import gymnasium as gym
import numpy as np
import torch

from wargame_rl.wargame.model.dqn.dqn import RL_Network
from wargame_rl.wargame.model.dqn.experience_replay import Experience, ReplayBuffer
from wargame_rl.wargame.model.dqn.state import state_to_tensor


class Agent:
    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer) -> None:
        """Base Agent class handling the interaction with the environment.

        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences

        """
        self.env = env
        self.replay_buffer = replay_buffer
        self.reset()
        self.state, _ = (
            self.env.reset()
        )  # this is a hack for now. TODO: define properly the state

    def reset(self) -> None:
        """Resents the environment and updates the state."""
        self.state, _ = self.env.reset()  # this is a hack for now

    def get_action(self, net: RL_Network, epsilon: float) -> int:
        """Using the given network, decide what action to carry out.

        Uses an epsilon-greedy policy.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action

        """
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state = state_to_tensor(self.state, net.device)
            q_values = net(state)
            _, action = torch.max(q_values, dim=1)

        return action.item()

    @torch.no_grad()
    def play_step(
        self,
        net: RL_Network,
        epsilon: float = 0.0,
    ) -> Tuple[float, bool]:
        """Carries out a single interaction step.

        Single interaction step between the agent and the environment.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, done

        """
        action = self.get_action(net, epsilon)

        # do step in the environment
        # So, in the deprecated version of gym, the env.step() has 4 values
        # unpacked which is: obs, reward, done, info = env.step(action)
        # In the latest version of gym, the step() function returns back an
        # additional variable which is truncated.
        #     obs, reward, terminated, truncated, info = env.step(action)
        new_state, reward, done, _, _ = self.env.step(action)

        exp = Experience(self.state, action, reward, done, new_state)

        self.replay_buffer.append(exp)

        self.state = new_state
        if done:
            self.reset()
        return reward, done
