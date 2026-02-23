from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvObservation
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.observation import observation_to_flat_tensor

if TYPE_CHECKING:
    from wargame_rl.wargame.model.ppo.ppo import PPOModel


class Experience:
    """Experience class to store state, action, reward, done, and log probability."""

    def __init__(
        self,
        state: Tensor,
        action: int,
        reward: float,
        done: bool,
        log_prob: Tensor,
    ) -> None:
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done
        self.log_prob = log_prob


class Agent:
    """Agent that interacts with the environment and collects experiences."""

    def __init__(self, env: WargameEnv) -> None:
        self.env = env

    def run_episode(
        self,
        policy_net: PPOModel,
        epsilon: float = 0.0,
        render: bool = False,
        save_steps: bool = True,
    ) -> tuple[float, int, list[Experience]]:
        """Run a single episode and return reward, steps, and optionally experiences.

        Args:
            policy_net: Policy network to use for action selection
            epsilon: Exploration rate (0 for deterministic)
            render: Whether to render the environment
            save_steps: Whether to save the steps as experiences

        Returns:
            (total_reward, steps, experiences) with experiences empty when save_steps is False
        """
        # Reset the environment
        observation: WargameEnvObservation
        observation, _ = self.env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        experiences = []

        # Convert observation to flat tensor (same layout as observation.size)
        state_tensor = observation_to_flat_tensor(observation, policy_net.device)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        while not done:
            # Get action from policy (single int); env expects one action per model
            action, log_prob = policy_net.get_action(
                state_tensor, deterministic=epsilon == 0.0
            )
            n_models = observation.n_wargame_models
            env_action = WargameEnvAction(actions=[action] * n_models)

            next_observation, reward, done, _, _ = self.env.step(env_action)
            total_reward += reward
            steps += 1

            log_prob_scalar = (
                log_prob.squeeze(0)
                if log_prob.dim() > 0 and log_prob.numel() > 1
                else log_prob
            )

            if save_steps:
                experiences.append(
                    Experience(
                        state=state_tensor.squeeze(0),
                        action=action,
                        reward=reward,
                        done=done,
                        log_prob=log_prob_scalar,
                    )
                )

            observation = next_observation
            state_tensor = observation_to_flat_tensor(observation, policy_net.device)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)

            # Render if requested
            if render:
                self.env.render()

        return total_reward, steps, experiences
