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
        hidden_dim=16,
        num_layers=1,
        device: Device = None,
    ):
        super(DQN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output = nn.Linear(hidden_dim, action_dim * n_wargame_models)
        self.populate_perfect_weights()
        self.activation = nn.ReLU()
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
        # don't load checkpoint because we're constructing the perfect weights
        # net.load_state_dict(state_dict)
        return net

    def populate_perfect_weights(self):
        # create a network which computes the l1 distance between the agent and the target
        # the l1 distance is the sum of the absolute differences in x and y coordinates (a - t).abs().sum()
        # abs can be represented with a relu function |a - t| = relu(a - t) + relu(t - a)
        abs_difference_weight = torch.as_tensor(
            [
                [0, 1, 0, -1, 0],  # agentX - targetX
                [0, -1, 0, 1, 0],  # targetX - agentX
                [0, 0, 1, 0, -1],  # agentY - targetY
                [0, 0, -1, 0, 1],  # targetY - agentY
            ],
            dtype=torch.float,
        )
        abs_difference_weight = abs_difference_weight.repeat(
            4, 1
        )  # repeat for each action

        step_size = 2 / 50
        # adjust the distance for each action
        # |(a + act) - t| = relu((a + act) - t) + relu(t - (a + act)) = relu(a - t + act) + relu(t - a - act)
        # fmt: off
        action_bias = torch.as_tensor(
            [
                step_size, -step_size, 0, 0,  # right
                0, 0, -step_size, step_size,  # up
                -step_size, step_size, 0, 0,  # left
                0, 0, step_size, -step_size,  # down
            ]
        )
        # fmt: on

        # sum up relu(a - t + act) + relu(t - a - act) and scale
        sum_weight = -0.25 * torch.eye(4).repeat_interleave(repeats=4, dim=1)

        with torch.no_grad():
            self.layers[0].weight[:] = abs_difference_weight  # shape 16x5
            self.layers[0].bias[:] = action_bias  # shape 16
            self.output.weight[:] = sum_weight  # shape 4x16
            self.output.bias.zero_()  # shape 4


def convert_state_dict(state_dict: dict) -> dict:
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("policy_net."):
            new_key = key[11:]
            new_state_dict[new_key] = value
    return new_state_dict
