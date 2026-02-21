from __future__ import annotations

from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.ppo.config import PPOConfig
from wargame_rl.wargame.model.ppo.networks import (
    PPOModel,
    PPOPolicyNetwork,
    PPOValueNetwork,
)

if TYPE_CHECKING:
    from wargame_rl.wargame.model.ppo.config import PPOConfig


class PPO_MLP(PPOModel):
    """PPO MLP Network for the wargame environment."""

    @classmethod
    def from_env(cls, env: WargameEnv, config: PPOConfig | None = None) -> PPO_MLP:
        """Create PPO_MLP from environment.

        Args:
            env: Wargame environment
            config: PPO configuration

        Returns:
            PPO_MLP instance
        """
        if config is None:
            config = PPOConfig()

        # Get input and output sizes from environment
        observation, _ = env.reset()
        obs_size = observation.size
        n_actions = env._action_handler.n_actions

        policy_network = PPOPolicyNetwork(
            input_size=obs_size,
            output_size=n_actions,
            device=env.device if hasattr(env, "device") else None,
        )
        value_network = PPOValueNetwork(
            input_size=obs_size,
            device=env.device if hasattr(env, "device") else None,
        )

        return cls(
            policy_network=policy_network,
            value_network=value_network,
            device=env.device if hasattr(env, "device") else None,
        )


class PPO_Transformer(PPOModel):
    """PPO Transformer Network for the wargame environment."""

    @classmethod
    def from_env(
        cls, env: WargameEnv, config: PPOConfig | None = None
    ) -> PPO_Transformer:
        """Create PPO_Transformer from environment.

        Args:
            env: Wargame environment
            config: PPO configuration

        Returns:
            PPO_Transformer instance
        """
        if config is None:
            config = PPOConfig()

        # Get input and output sizes from environment
        observation, _ = env.reset()
        objective_size = observation.size_objectives[0]
        wargame_model_size = observation.size_wargame_models[0]
        game_size = observation.size_game_observation
        n_actions = env._action_handler.n_actions

        # For transformer, we'll need to define the transformer architecture
        # This is a placeholder implementation - we'll use the same MLP approach for now
        # but with proper sizes
        policy_network = PPOPolicyNetwork(
            input_size=objective_size + wargame_model_size + game_size,
            output_size=n_actions,
            device=env.device if hasattr(env, "device") else None,
        )
        value_network = PPOValueNetwork(
            input_size=objective_size + wargame_model_size + game_size,
            device=env.device if hasattr(env, "device") else None,
        )

        return cls(
            policy_network=policy_network,
            value_network=value_network,
            device=env.device if hasattr(env, "device") else None,
        )
