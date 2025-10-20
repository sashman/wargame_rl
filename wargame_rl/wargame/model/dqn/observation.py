import torch
from wandb.util import np

from wargame_rl.wargame.envs.types import (
    WargameEnvAction,
    WargameEnvObjectiveObservation,
    WargameEnvObservation,
    WargameModelObservation,
)
from wargame_rl.wargame.model.dqn.device import Device, get_device


def action_to_tensor(action: WargameEnvAction, device: Device = None) -> torch.Tensor:
    device = get_device(device)
    _, action_tensor = torch.tensor(
        action.actions, dtype=torch.float32, device=device
    ).max(axis=-1)
    return action_tensor.unsqueeze(0)


def normalize_location(location: np.ndarray) -> np.ndarray:
    half_grid_size = 25
    return (location - half_grid_size) / half_grid_size


def normalize_distances(distances: np.ndarray) -> np.ndarray:
    # Maximum possible distance in a 50x50 grid is diagonal distance
    # sqrt(50^2 + 50^2) = sqrt(5000) ≈ 70.7
    half_max_distance = (np.sqrt(2) * 50) / 2  # diagonal distance
    return (distances - half_max_distance) / half_max_distance


def observation_to_tensor(
    state: WargameEnvObservation, device: Device = None
) -> torch.Tensor:
    device = get_device(device)

    # current_turn: int = state.current_turn
    wargame_models: list[WargameModelObservation] = state.wargame_models
    objectives: list[WargameEnvObjectiveObservation] = state.objectives
    # tensor_current_turn = torch.tensor(
    #     [current_turn], dtype=torch.float32, device=device
    # )
    tensor_current_turn = torch.tensor([0], dtype=torch.float32, device=device)
    tensor_wargame_models = torch.tensor(
        [
            [
                normalize_location(model.location),
                normalize_distances(model.distances_to_objectives),
            ]
            for model in wargame_models
        ],
        dtype=torch.float32,
        device=device,
    )
    tensor_objectives = torch.tensor(
        [normalize_location(objective.location) for objective in objectives],
        dtype=torch.float32,
        device=device,
    )
    tensor_state = torch.cat(
        [
            tensor_current_turn,
            tensor_wargame_models.flatten(),
            tensor_objectives.flatten(),
        ],
        dim=0,
    )

    return tensor_state.unsqueeze(0)


def observations_to_tensor_batch(
    states: list[WargameEnvObservation], device: Device = None
) -> torch.Tensor:
    device = get_device(device)

    return torch.cat([observation_to_tensor(state, device) for state in states], dim=0)
