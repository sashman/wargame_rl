from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvObservation
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.dqn.experience_replay import ReplayBuffer
from wargame_rl.wargame.types import Experience


class BaseAgent(ABC):
    """Common interface for RL agents interacting with the environment."""

    def __init__(self, env: WargameEnv) -> None:
        self.env = env
        self.observation: WargameEnvObservation | None = None
        self._last_log_prob: torch.Tensor | None = None
        self.replay_buffer: ReplayBuffer | None = None
        self.last_episode_reward_breakdown: dict[str, float] = {}

    def reset(self) -> None:
        """Reset environment state for a new episode."""
        self.observation, _ = self.env.reset()
        self._last_log_prob = None

    def _require_observation(self) -> WargameEnvObservation:
        if self.observation is None:
            self.reset()
        assert self.observation is not None
        return self.observation

    @abstractmethod
    def get_action(
        self, policy: Any, observation: WargameEnvObservation, epsilon: float
    ) -> WargameEnvAction:
        """Select an action."""

    def _record_experience(self, exp: Experience) -> None:
        """Hook for agents that store experiences (e.g., replay buffer)."""
        if self.replay_buffer is not None:
            self.replay_buffer.append(exp)

    @torch.no_grad()
    def play_step(
        self,
        policy: Any,
        epsilon: float = 0.0,
        save_step: bool = True,
    ) -> tuple[float, bool, Experience | None]:
        """Execute one env step and optionally return the Experience."""
        observation = self._require_observation()
        action = self.get_action(policy, observation, epsilon)
        log_prob = self._last_log_prob
        new_state, reward, done, _, _info = self.env.step(action)

        exp: Experience | None = None
        if save_step:
            exp = Experience(
                state=observation,
                action=action,
                reward=float(reward),
                done=bool(done),
                new_state=new_state,
                log_prob=log_prob,
            )
            self._record_experience(exp)

        self.observation = new_state
        return float(reward), bool(done), exp

    def run_episode(
        self,
        policy: Any,
        epsilon: float = 0.0,
        render: bool = False,
        save_steps: bool = True,
    ) -> tuple[float, int]:
        """Run a single episode and return (total_reward, steps)."""
        total_reward, steps, _experiences = self.run_episode_with_experiences(
            policy, epsilon=epsilon, render=render, save_steps=save_steps
        )
        return total_reward, steps

    def run_episode_with_experiences(
        self,
        policy: Any,
        epsilon: float = 0.0,
        render: bool = False,
        save_steps: bool = True,
    ) -> tuple[float, int, list[Experience]]:
        """Run a single episode and return (total_reward, steps, experiences)."""
        self.reset()
        total_reward = 0.0
        steps = 0
        done = False
        experiences: list[Experience] = []

        while not done:
            reward, done, exp = self.play_step(
                policy, epsilon=epsilon, save_step=save_steps
            )
            total_reward += reward
            steps += 1
            if exp is not None:
                experiences.append(exp)

            if render:
                self.env.render()

        if self.env.episode_reward_steps > 0:
            self.last_episode_reward_breakdown = {
                key: value / float(self.env.episode_reward_steps)
                for key, value in self.env.episode_reward_breakdown.items()
            }
        else:
            self.last_episode_reward_breakdown = {}

        return total_reward, steps, experiences
