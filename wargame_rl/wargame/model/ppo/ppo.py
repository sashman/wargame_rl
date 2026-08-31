from __future__ import annotations

from typing import TYPE_CHECKING, cast

from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.config import TransformerConfig
from wargame_rl.wargame.model.net import TransformerNetwork
from wargame_rl.wargame.model.ppo.config import PPOConfig
from wargame_rl.wargame.model.ppo.networks import PPOModel

if TYPE_CHECKING:
    from wargame_rl.wargame.model.ppo.config import PPOConfig


class PPO_Transformer(PPOModel):
    """PPO Transformer Network for the wargame environment."""

    @classmethod
    def from_env(
        cls,
        env: WargameEnv,
        config: PPOConfig | None = None,
        transformer_config: TransformerConfig | None = None,
    ) -> PPO_Transformer:
        """Create PPO_Transformer from environment.

        Args:
            env: Wargame environment
            config: PPO configuration
            transformer_config: trunk depth and width. `None` is the production
                size and is bit-identical to before this parameter existed; see
                `TransformerNetwork.from_spec` for why it is a parameter and why
                a network built at another size is a different network.

        Returns:
            PPO_Transformer instance
        """
        if config is None:
            config = PPOConfig()

        # Both halves take the SAME trunk config. They are separate networks
        # unless `share_transformer`, but a policy and a value head of different
        # widths is not a configuration anything here wants, and passing it once
        # is what stops them drifting apart.
        policy_network = TransformerNetwork.from_env(
            env=env, is_policy=True, transformer_config=transformer_config
        )
        value_network = TransformerNetwork.from_env(
            env=env, is_policy=False, transformer_config=transformer_config
        )

        net = cls(
            policy_network=policy_network,
            value_network=value_network,
            device=env.device if hasattr(env, "device") else None,
            share_transformer=config.share_transformer,
        )
        return cast(PPO_Transformer, net.to(net.device))
