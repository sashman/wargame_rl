from __future__ import annotations

from typing import TYPE_CHECKING, cast

from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.net import TransformerNetwork
from wargame_rl.wargame.model.ppo.config import PPOConfig
from wargame_rl.wargame.model.ppo.networks import PPOModel

if TYPE_CHECKING:
    from wargame_rl.wargame.model.ppo.config import PPOConfig


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

        policy_network = TransformerNetwork.from_env(env=env, is_policy=True)
        value_network = TransformerNetwork.from_env(env=env, is_policy=False)

        net = cls(
            policy_network=policy_network,
            value_network=value_network,
            device=env.device if hasattr(env, "device") else None,
            share_transformer=config.share_transformer,
        )
        return cast(PPO_Transformer, net.to(net.device))
