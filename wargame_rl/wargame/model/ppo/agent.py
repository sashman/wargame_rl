from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from wargame_rl.wargame.envs.types import WargameEnvAction
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.agent_base import BaseAgent
from wargame_rl.wargame.model.common.observation import observation_to_tensor
from wargame_rl.wargame.types import Experience

if TYPE_CHECKING:
    from wargame_rl.wargame.model.ppo.ppo import PPOModel


class Agent(BaseAgent):
    """Agent that interacts with the environment and collects experiences."""

    def __init__(self, env: WargameEnv) -> None:
        super().__init__(env)

    @torch.no_grad()
    def get_action(
        self, policy_net: PPOModel, epsilon: float
    ) -> tuple[WargameEnvAction, torch.Tensor | None]:
        observation = self._require_observation()
        state_tensors = observation_to_tensor(observation, policy_net.device)
        env_action, log_prob = policy_net.get_action(
            state_tensors, deterministic=(epsilon == 0.0)
        )
        return env_action, log_prob

    @torch.no_grad()
    def play_step(
        self,
        policy_net: PPOModel,
        epsilon: float = 0.0,
        save_step: bool = True,
    ) -> tuple[float, bool, Experience | None]:
        observation = self._require_observation()
        env_action, log_prob = self.get_action(policy_net, epsilon)
        next_observation, reward, done, _, _ = self.env.step(env_action)

        exp: Experience | None = None
        if save_step:
            exp = Experience(
                state=observation,
                new_state=next_observation,
                action=env_action,
                reward=reward,
                done=done,
                log_prob=log_prob,
            )
        self.observation = next_observation
        return reward, done, exp
