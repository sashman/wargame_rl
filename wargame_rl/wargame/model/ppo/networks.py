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
            encoded, n_prefix, n_wargame_models = policy_network.encode_state(x)
            action_logits = policy_network.policy_from_encoded(
                encoded, n_prefix, n_wargame_models
            )
            state_values = value_network.value_from_encoded(encoded)
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
            (env_action, joint_log_prob) where env_action contains a per-model
            action list and joint_log_prob is the sum of per-model log-probs (scalar).
        """
        action_logits, _ = self.forward(state_tensors)
        action_dist = Categorical(logits=action_logits)

        if deterministic:
            actions = torch.argmax(action_logits, dim=-1)
        else:
            actions = action_dist.sample()

        per_model_log_probs = action_dist.log_prob(actions)
        joint_log_prob = per_model_log_probs.sum(dim=-1).squeeze(0)

        env_action = WargameEnvAction(actions=actions.flatten().tolist())
        return env_action, joint_log_prob

    def evaluate_actions(
        self, state_tensors: list[torch.Tensor], actions: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Evaluate actions under the current policy.

        Args:
            state_tensors: Batch of observations as tensors.
            actions: Per-model actions, shape (batch_size, n_models).

        Returns:
            (action_logits, joint_log_probs, joint_entropy) where the joint
            quantities are summed across models, giving shape (batch_size,).
        """
        action_logits, _ = self.forward(state_tensors)
        action_dist = Categorical(logits=action_logits)

        joint_log_probs = action_dist.log_prob(actions).sum(dim=-1)
        joint_entropy = action_dist.entropy().sum(dim=-1)

        return action_logits, joint_log_probs, joint_entropy
