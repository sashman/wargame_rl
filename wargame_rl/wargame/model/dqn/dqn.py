from abc import ABC, abstractmethod
import torch
from torch import nn
import torch.nn.functional as F
from wargame_rl.wargame.model.dqn.device import Device, get_device
from wargame_rl.wargame.model.dqn.state import state_to_tensor
from wargame_rl.wargame.types import State


class RL_Network(nn.Module, ABC):
    device: torch.device

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class DQN(RL_Network):
    def __init__(
        self, state_dim, action_dim, hidden_dim=256, num_layers=2, device: Device = None
    ):
        super(DQN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output = nn.Linear(hidden_dim, action_dim)
        self.activation = nn.GELU()
        self.device = get_device(device)
        self.to(self.device)

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.output(x)
