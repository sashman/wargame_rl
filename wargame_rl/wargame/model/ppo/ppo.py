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

        # Only hand the slice over when the flag is on, so `shooting_slice`
        # being set is by itself the switch the decode path tests.
        shooting_slice: tuple[int, int] | None = None
        if config.distinct_shooting_targets:
            env_slice = env.player_action_handler.shooting_slice
            if env_slice is None:
                raise ValueError(
                    "distinct_shooting_targets requires a config whose models "
                    "carry weapons; this env registered no shooting actions."
                )
            shooting_slice = (env_slice.start, env_slice.end)

        net = cls(
            policy_network=policy_network,
            value_network=value_network,
            device=env.device if hasattr(env, "device") else None,
            share_transformer=config.share_transformer,
            shooting_slice=shooting_slice,
        )
        return cast(PPO_Transformer, net.to(net.device))
