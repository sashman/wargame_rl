from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical

from wargame_rl.wargame.model.common import Device, get_device

if TYPE_CHECKING:
    pass


class PPOPolicyNetwork(nn.Module):
    """PPO Policy Network that outputs action probabilities."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        device: Device = None,
    ) -> None:
        super().__init__()
        self._device = get_device(device)

        layers: list[nn.Module] = []
        prev_size = input_size

        # Create hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Softmax(dim=-1))

        self.network = nn.Sequential(*layers)

    @property
    def device(self) -> torch.device:
        return self._device

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the policy network.

        Args:
            x: Input tensor

        Returns:
            Action probabilities
        """
        out: Tensor = self.network(x)
        return out


class PPOValueNetwork(nn.Module):
    """PPO Value Network that estimates state values."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        device: Device = None,
    ) -> None:
        super().__init__()
        self._device = get_device(device)

        layers_val: list[nn.Module] = []
        prev_size = input_size

        # Create hidden layers
        for _ in range(num_layers - 1):
            layers_val.append(nn.Linear(prev_size, hidden_size))
            layers_val.append(nn.ReLU())
            prev_size = hidden_size

        # Output layer
        layers_val.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers_val)

    @property
    def device(self) -> torch.device:
        return self._device

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the value network.

        Args:
            x: Input tensor

        Returns:
            State values
        """
        out: Tensor = self.network(x).squeeze(-1)
        return out


class PPOModel(nn.Module):
    """PPO Model combining policy and value networks."""

    def __init__(
        self,
        policy_network: PPOPolicyNetwork,
        value_network: PPOValueNetwork,
        device: Device = None,
    ) -> None:
        super().__init__()
        self.policy_network = policy_network
        self.value_network = value_network
        self._device = get_device(device)

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
