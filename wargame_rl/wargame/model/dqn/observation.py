import numpy as np
import torch
from torch import Tensor

from wargame_rl.wargame.envs.types import (
    WargameEnvAction,
    WargameEnvObjectiveObservation,
    WargameEnvObservation,
    WargameModelObservation,
)
from wargame_rl.wargame.model.dqn.device import Device, get_device


def action_to_tensor(action: WargameEnvAction, device: Device | None = None) -> Tensor:
    device = get_device(device)
    _, action_tensor = torch.tensor(
        action.actions, dtype=torch.float32, device=device
    ).max(dim=-1)
    return action_tensor.unsqueeze(0)


def normalize_location(location: np.ndarray) -> np.ndarray:
    half_grid_size = 25
    return (location - half_grid_size) / half_grid_size


def normalize_distances(distances: np.ndarray) -> np.ndarray:
    # Maximum possible distance in a 50x50 grid is diagonal distance
    # sqrt(50^2 + 50^2) = sqrt(5000) ≈ 70.7
    half_max_distance = (np.sqrt(2) * 50) / 2  # diagonal distance
    return np.array((distances - half_max_distance) / half_max_distance)


def group_id_to_one_hot(group_id: int, max_groups: int) -> list[float]:
    """Encode positive group_id as one-hot vector of length max_groups. Clamps to valid index."""
    index = min(max(0, group_id - 1), max_groups - 1)
    return [1.0 if i == index else 0.0 for i in range(max_groups)]


def observation_to_tensor(
    state: WargameEnvObservation, device: Device | None = None
) -> list[torch.Tensor]:
    """Convert observation to tensors.

    Order of tensors
    ----------------

    The tensors are returned in the following order:
        1. tensor_current_turn: Tensor containing the current turn as a float32
        tensor of shape (1,).
        2. tensor_objectives: Tensor of all objectives; shape (num_objectives, location_dims),
        values normalized to [-1, 1].
        3. tensor_wargame_models: Tensor of all wargame models; shape (num_models, model_features),
        where model_features includes normalized location, distances to objectives (normalized to [-1, 1]),
        and group_id as one-hot of length max_groups.
    """

    device = get_device(device)

    # current_turn: int = state.current_turn
    wargame_models: list[WargameModelObservation] = state.wargame_models
    objectives: list[WargameEnvObjectiveObservation] = state.objectives
    # tensor_current_turn = torch.tensor(
    #     [current_turn], dtype=torch.float32, device=device
    # )
    tensor_current_turn: Tensor = torch.tensor([0], dtype=torch.float32, device=device)
    max_groups: int = state.wargame_models[0].max_groups
    tensor_wargame_models: Tensor = torch.tensor(
        [
            [
                *normalize_location(model.location),
                *normalize_location(model.distances_to_objectives.flatten()),
                *group_id_to_one_hot(model.group_id, max_groups),
            ]
            for model in wargame_models
        ],
        dtype=torch.float32,
        device=device,
    )
    tensor_objectives: Tensor = torch.tensor(
        [normalize_location(objective.location) for objective in objectives],
        dtype=torch.float32,
        device=device,
    )
    return [
        tensor_current_turn,
        tensor_objectives,
        tensor_wargame_models,
    ]


def observations_to_tensor_batch(
    states: list[WargameEnvObservation], device: Device = None
) -> list[torch.Tensor]:
    assert len(states) > 0, "No states to convert to tensor"
    device = get_device(device)

    tensors = [observation_to_tensor(state, device) for state in states]

    tensor_current_turn = torch.cat(
        [tensor[0].unsqueeze(0) for tensor in tensors], dim=0
    )
    tensor_objectives = torch.cat([tensor[1].unsqueeze(0) for tensor in tensors], dim=0)
    tensor_wargame_models = torch.cat(
        [tensor[2].unsqueeze(0) for tensor in tensors], dim=0
    )

    return [tensor_current_turn, tensor_objectives, tensor_wargame_models]
