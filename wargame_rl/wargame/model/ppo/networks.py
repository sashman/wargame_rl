from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical

from wargame_rl.wargame.model.common import Device, get_device
from wargame_rl.wargame.model.net import RL_Network

if TYPE_CHECKING:
    pass


class PPOModel(nn.Module):
    """PPO Model combining policy and value networks."""

    def __init__(
        self,
        policy_network: RL_Network,
        value_network: RL_Network,
        device: Device = None,
    ) -> None:
        super().__init__()
        if not policy_network.is_policy or value_network.is_policy:
            raise ValueError("Wrong network type.")
        self.policy_network = policy_network
        self.value_network = value_network
        self._device = get_device(device)
        self.to(self.device)

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: Device) -> PPOModel:  # type: ignore[override]
        """Move the model to the specified device."""
        self._device = get_device(device)
        self.policy_network = self.policy_network.to(device)
        self.value_network = self.value_network.to(device)
        return self

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass through both networks.

        Args:
            x: Input tensor

        Returns:
            (action_probabilities, state_values)
        """
        action_probs = self.policy_network(x)
        state_values = self.value_network(x)
        return action_probs, state_values

    def get_action(
        self, state: Tensor, deterministic: bool = False
    ) -> Tuple[int, Tensor]:
        """Get action from policy network.

        Args:
            state: Input state tensor
            deterministic: Whether to sample or take the argmax

        Returns:
            (action, log_prob)
        """
        action_probs, _ = self.forward(state)
        action_dist = Categorical(action_probs)
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            action = action_dist.sample()

        log_prob = action_dist.log_prob(action)
        return int(action.item()), log_prob

    def evaluate_actions(
        self, state: Tensor, actions: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Evaluate actions under the current policy.

        Args:
            state: Input state tensor
            actions: Actions to evaluate

        Returns:
            (action_probabilities, log_probabilities, entropy)
        """
        action_probs, _ = self.forward(state)
        action_dist = Categorical(action_probs)

        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()

        return action_probs, log_probs, entropy
