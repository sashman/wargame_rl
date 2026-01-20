from typing import Tuple

import gymnasium as gym
import numpy as np
import torch

from wargame_rl.wargame.envs.types import WargameEnvAction
from wargame_rl.wargame.model.dqn.dqn import RL_Network
from wargame_rl.wargame.model.dqn.experience_replay import Experience, ReplayBuffer
from wargame_rl.wargame.model.dqn.observation import observation_to_tensor


class Agent:
    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer | None = None) -> None:
        """Base Agent class handling the interaction with the environment.

        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences

        """
        self.env = env
        self.replay_buffer = replay_buffer
        self.reset()
        self.observation, info = self.env.reset()

    def reset(self) -> None:
        """Resents the environment and updates the state."""
        self.observation, _ = self.env.reset()  # this is a hack for now

    def get_action(self, policy_net: RL_Network, epsilon: float) -> WargameEnvAction:
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
            action = WargameEnvAction(self.env.action_space.sample())
        else:
            with torch.no_grad():
                state = observation_to_tensor(self.observation, policy_net.device)
                q_values = policy_net(state)
                assert q_values.shape[0] == 1
                assert len(q_values.shape) == 3
                _, action_indexes = q_values.max(axis=-1)
                action = WargameEnvAction(actions=action_indexes.flatten().tolist())

        return action

    @torch.no_grad()
    def play_step(
        self,
        net: RL_Network,
        epsilon: float = 0.0,
        save_step: bool = True,
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

        if self.replay_buffer is not None and save_step:
            exp = Experience(
                state=self.observation,
                action=action,
                reward=float(reward),
                done=bool(done),
                new_state=new_state,
            )
            self.replay_buffer.append(exp)
        self.observation = new_state
        return float(reward), bool(done)

    def run_episode(
        self,
        net: RL_Network,
        epsilon: float = 0.0,
        render: bool = False,
        save_steps: bool = True,
    ) -> tuple[float, int]:
        """Run a single episode with the trained agent.

        Args:
            net: DQN model
            epsilon: value to determine likelihood of taking a random action
            render: Whether to render the environment
            save_steps: Whether to save the steps to the replay buffer

        Returns:
            Total reward and number of steps taken
        """

        self.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            reward, done = self.play_step(net, epsilon, save_step=save_steps)
            total_reward += reward
            steps += 1

            if render:
                self.env.render()

        return total_reward, steps
