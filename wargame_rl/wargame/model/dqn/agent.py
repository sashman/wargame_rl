from typing import Tuple

import gymnasium as gym
import numpy as np
import torch

from wargame_rl.wargame.envs.types import WargameEnvAction
from wargame_rl.wargame.model.common.agent_base import BaseAgent
from wargame_rl.wargame.model.common.observation import (
    apply_action_mask,
    observation_to_tensor,
)
from wargame_rl.wargame.model.dqn.experience_replay import ReplayBuffer
from wargame_rl.wargame.model.net import RL_Network
from wargame_rl.wargame.types import Experience


class Agent(BaseAgent):
    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer | None = None) -> None:
        """Base Agent class handling the interaction with the environment.

        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences

        """
        super().__init__(env)
        self.replay_buffer = replay_buffer
        self.reset()

    def get_action(
        self, policy_net: RL_Network, epsilon: float
    ) -> tuple[WargameEnvAction, None]:
        """Using the given network, decide what action to carry out.

        Uses an epsilon-greedy policy with action masking — only valid
        actions (according to ``observation.action_mask``) are considered.
        """
        observation = self._require_observation()
        mask = observation.action_mask  # (n_models, n_actions) or None

        if np.random.random() < epsilon:
            if mask is not None:
                action = WargameEnvAction.random(mask)
            else:
                action = WargameEnvAction(self.env.action_space.sample())
        else:
            with torch.no_grad():
                tensors = observation_to_tensor(observation, policy_net.device)
                mask_tensor = tensors[4]  # (n_models, n_actions)
                state = tensors[:4]
                q_values = policy_net(state)
                assert q_values.shape[0] == 1
                assert len(q_values.shape) == 3
                q_values = apply_action_mask(q_values, mask_tensor.unsqueeze(0))
                _, action_indexes = q_values.max(dim=-1)
                action = WargameEnvAction(actions=action_indexes.flatten().tolist())

        return action, None

    @torch.no_grad()
    def play_step(
        self,
        net: RL_Network,
        epsilon: float = 0.0,
        save_step: bool = True,
    ) -> Tuple[float, bool, Experience | None]:
        """Carries out a single interaction step.

        Single interaction step between the agent and the environment.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, done

        """
        action, _log_prob = self.get_action(net, epsilon)

        # do step in the environment
        # So, in the deprecated version of gym, the env.step() has 4 values
        # unpacked which is: obs, reward, done, info = env.step(action)
        # In the latest version of gym, the step() function returns back an
        # additional variable which is truncated.
        #     obs, reward, terminated, truncated, info = env.step(action)
        observation = self._require_observation()
        new_state, reward, done, _, _info = self.env.step(action)

        exp: Experience | None = None
        if save_step:
            exp = Experience(
                state=observation,
                action=action,
                reward=float(reward),
                done=bool(done),
                new_state=new_state,
                log_prob=None,
            )
            if self.replay_buffer is not None:
                self.replay_buffer.append(exp)
        self.observation = new_state
        return float(reward), bool(done), exp
