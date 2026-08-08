from __future__ import annotations

from typing import Tuple, cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical

from wargame_rl.wargame.envs.types import WargameEnvAction
from wargame_rl.wargame.model.common import Device, get_device
from wargame_rl.wargame.model.net import RL_Network, TransformerNetwork


class PPOModel(nn.Module):
    """PPO Model combining policy and value networks."""

    def __init__(
        self,
        policy_network: RL_Network,
        value_network: RL_Network,
        device: Device = None,
        share_transformer: bool = False,
    ) -> None:
        super().__init__()
        if not policy_network.is_policy or value_network.is_policy:
            raise ValueError("Wrong network type.")
        if share_transformer and (
            not isinstance(policy_network, TransformerNetwork)
            or not isinstance(value_network, TransformerNetwork)
        ):
            raise ValueError(
                "`share_transformer=True` requires TransformerNetwork for both policy and value networks."
            )
        if share_transformer:
            assert isinstance(policy_network, TransformerNetwork)
            assert isinstance(value_network, TransformerNetwork)
            value_network.share_backbone_with(policy_network)
        self.policy_network = policy_network
        self.value_network = value_network
        self.share_transformer = share_transformer
        self.to(get_device(device))

    @property
    def device(self) -> torch.device:  # type: ignore[override]
        """Derive device from actual parameter location (stays correct after Lightning moves the model)."""
        param = next(self.parameters(), None)
        if param is not None:
            return param.device
        return torch.device("cpu")

    def forward(self, x: list[torch.Tensor]) -> Tuple[Tensor, Tensor]:
        """Forward pass through both networks.

        Args:
            x: List of input tensors (game state, objectives, models).

        Returns:
            (action_logits, state_values) where action_logits has shape
            (batch, n_models, n_actions) and state_values has shape (batch,).
        """
        if self.share_transformer:
            policy_network = cast(TransformerNetwork, self.policy_network)
            value_network = cast(TransformerNetwork, self.value_network)
            encoded_state = policy_network.encode_state(x)
            action_logits = policy_network.policy_from_encoded(encoded_state)
            state_values = value_network.value_from_encoded(encoded_state)
            return action_logits, state_values

        action_logits = self.policy_network(x)
        state_values = self.value_network(x)
        return action_logits, state_values

    def get_action(
        self, state_tensors: list[torch.Tensor], deterministic: bool = False
    ) -> Tuple[WargameEnvAction, Tensor]:
        """Select one action per model, mirroring how DQN selects per-model actions.

        Args:
            state_tensors: Observation converted to tensors (single observation, not batched).
            deterministic: If True take argmax, otherwise sample.

        Returns:
            (env_action, per_model_log_probs) where env_action contains a
            per-model action list and per_model_log_probs has shape
            (n_models,) — kept unsummed so PPO can clip each model's
            importance ratio separately.
        """
        action_logits, _ = self.forward(state_tensors)
        action_dist = Categorical(logits=action_logits)

        if deterministic:
            actions = torch.argmax(action_logits, dim=-1)
        else:
            actions = action_dist.sample()

        per_model_log_probs = action_dist.log_prob(actions).squeeze(0)

        env_action = WargameEnvAction(actions=actions.flatten().tolist())
        return env_action, per_model_log_probs

    def evaluate_actions(
        self, state_tensors: list[torch.Tensor], actions: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Evaluate actions under the current policy.

        Args:
            state_tensors: Batch of observations as tensors.
            actions: Per-model actions, shape (batch_size, n_models).

        Returns:
            (action_logits, log_probs, entropy) with the per-model quantities
            left unsummed, shape (batch_size, n_models).
        """
        action_logits, _ = self.forward(state_tensors)
        action_dist = Categorical(logits=action_logits)

        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()

        return action_logits, log_probs, entropy
