from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvObservation
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.agent_base import BaseAgent
from wargame_rl.wargame.model.common.observation import observation_to_tensor

if TYPE_CHECKING:
    from wargame_rl.wargame.model.ppo.ppo import PPOModel


class Agent(BaseAgent):
    """Agent that interacts with the environment and collects experiences."""

    def __init__(self, env: WargameEnv) -> None:
        super().__init__(env)

    @torch.no_grad()
    def get_action(
        self,
        policy_net: PPOModel,
        observation: WargameEnvObservation,
        epsilon: float,
    ) -> WargameEnvAction:
        state_tensors = observation_to_tensor(observation, policy_net.device)
        env_action, log_prob = policy_net.get_action(
            state_tensors, deterministic=(epsilon == 0.0)
        )
        self._last_log_prob = log_prob
        return env_action
