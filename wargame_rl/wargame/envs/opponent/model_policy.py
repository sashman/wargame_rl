"""Opponent policy that uses a frozen PPO policy checkpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from wargame_rl.wargame.envs.opponent.perspective import build_opponent_observation
from wargame_rl.wargame.envs.opponent.policy import OpponentPolicy
from wargame_rl.wargame.envs.opponent.registry import register_policy
from wargame_rl.wargame.envs.types import WargameEnvAction
from wargame_rl.wargame.model.common.observation import (
    apply_action_mask,
    observation_to_tensor,
)
from wargame_rl.wargame.model.common.policy_checkpoint import load_ppo_policy_state_dict

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.wargame import WargameEnv
    from wargame_rl.wargame.envs.wargame_model import WargameModel


class ModelPolicy(OpponentPolicy):
    """Load a PPO policy snapshot/checkpoint and use it as an opponent."""

    def __init__(
        self,
        env: WargameEnv,
        checkpoint_path: str,
        deterministic: bool = True,
        **kwargs: object,
    ) -> None:
        del kwargs
        self._env = env
        self._deterministic = deterministic
        from wargame_rl.wargame.model.common.config import TransformerConfig
        from wargame_rl.wargame.model.net import TransformerNetwork

        bootstrap_obs = build_opponent_observation(env, action_mask=None)
        objective_size = bootstrap_obs.size_objectives[0]
        model_size = bootstrap_obs.size_wargame_models[0]
        game_size = bootstrap_obs.size_game_observation
        opponent_model_size = (
            bootstrap_obs.size_opponent_models[0]
            if bootstrap_obs.size_opponent_models
            else 0
        )

        self._policy = TransformerNetwork(
            game_size=game_size,
            objective_size=objective_size,
            wargame_model_size=model_size,
            n_actions=env._opponent_action_handler.n_actions,
            is_policy=True,
            transformer_config=TransformerConfig(),
            opponent_model_size=opponent_model_size,
        )
        state_dict = load_ppo_policy_state_dict(checkpoint_path)
        self._policy.load_state_dict(state_dict)
        self._policy.eval()

    @torch.no_grad()
    def select_action(
        self,
        opponent_models: list[WargameModel],
        env: WargameEnv,
        action_mask: np.ndarray | None = None,
    ) -> WargameEnvAction:
        if len(opponent_models) == 0:
            return WargameEnvAction(actions=[])

        obs = build_opponent_observation(env, action_mask=action_mask)
        tensors = observation_to_tensor(obs, self._policy.device)
        logits = self._policy(tensors)

        if action_mask is not None:
            mask_tensor = tensors[4]
            logits = apply_action_mask(logits, mask_tensor.unsqueeze(0))

        if self._deterministic:
            action_indexes = torch.argmax(logits, dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action_indexes = dist.sample()

        return WargameEnvAction(actions=action_indexes.flatten().tolist())


register_policy("model", ModelPolicy)
