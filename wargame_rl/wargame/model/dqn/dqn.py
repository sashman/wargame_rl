from abc import ABC, abstractmethod
from typing import Self

import gymnasium as gym
import torch
from torch import nn

from wargame_rl.wargame.envs.wargame import MovementPhaseActions
from wargame_rl.wargame.model.dqn.device import Device, get_device


class RL_Network(nn.Module, ABC):
    device: torch.device

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class DQN(RL_Network):
    def __init__(
        self,
        state_dim,
        action_dim,
        n_wargame_models,
        hidden_dim=256,
        num_layers=3,
        device: Device = None,
    ):
        super(DQN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output = nn.Linear(hidden_dim, action_dim * n_wargame_models)
        self.activation = nn.GELU()
        self.device = get_device(device)
        self.to(self.device)
        self.n_wargame_models = n_wargame_models
        self.action_dim = action_dim

    def forward(self, x):
        assert len(x.shape) == 2
        batch_size = x.shape[0]
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.output(x)
        return x.reshape(batch_size, self.n_wargame_models, self.action_dim)

    @classmethod
    def from_env(cls, env: gym.Env) -> "DQN":
        observation, _ = env.reset()
        obs_size = observation.size
        n_wargame_models = observation.n_wargame_models
        n_actions = len(MovementPhaseActions)
        return cls(obs_size, n_actions, n_wargame_models)

    @classmethod
    def from_checkpoint(cls, env: gym.Env, checkpoint_path: str) -> Self:
        load_dict = torch.load(checkpoint_path, weights_only=False)
        if "state_dict" in load_dict:
            state_dict = convert_state_dict(load_dict["state_dict"])
        else:
            state_dict = load_dict
        return cls.from_state_dict(env, state_dict)

    @classmethod
    def from_state_dict(cls, env: gym.Env, state_dict: dict) -> Self:
        net = cls.from_env(env)
        net.load_state_dict(state_dict)
        return net


def convert_state_dict(state_dict: dict) -> dict:
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("policy_net."):
            new_key = key[11:]
            new_state_dict[new_key] = value
    return new_state_dict
