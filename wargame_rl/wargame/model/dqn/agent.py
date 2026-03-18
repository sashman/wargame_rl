import gymnasium as gym
import numpy as np
import torch

from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvObservation
from wargame_rl.wargame.model.common.agent_base import BaseAgent
from wargame_rl.wargame.model.common.observation import (
    apply_action_mask,
    observation_to_tensor,
)
from wargame_rl.wargame.model.dqn.experience_replay import ReplayBuffer
from wargame_rl.wargame.model.net import RL_Network


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
        self,
        policy_net: RL_Network,
        observation: WargameEnvObservation,
        epsilon: float,
    ) -> WargameEnvAction:
        """Using the given network, decide what action to carry out.

        Uses an epsilon-greedy policy with action masking — only valid
        actions (according to ``observation.action_mask``) are considered.
        """
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

        self._last_log_prob = None
        return action
