from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvObservation
from wargame_rl.wargame.types import Experience


class BaseAgent(ABC):
    """Common interface for RL agents interacting with the environment."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.observation: WargameEnvObservation | None = None

    def reset(self) -> None:
        """Reset environment state for a new episode."""
        self.observation, _ = self.env.reset()

    def _require_observation(self) -> WargameEnvObservation:
        if self.observation is None:
            self.reset()
        assert self.observation is not None
        return self.observation

    @abstractmethod
    def get_action(self, policy: Any, epsilon: float) -> tuple[WargameEnvAction, Any]:
        """Select an action (and optional metadata like log-prob)."""

    @abstractmethod
    def play_step(
        self,
        policy: Any,
        epsilon: float = 0.0,
        save_step: bool = True,
    ) -> tuple[float, bool, Experience | None]:
        """Execute one env step and optionally return the Experience."""

    def run_episode(
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

        return total_reward, steps, experiences
