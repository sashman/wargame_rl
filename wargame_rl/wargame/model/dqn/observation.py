import torch

from wargame_rl.wargame.envs.env_types import (
    WargameEnvObjectiveObservation,
    WargameEnvObservation,
    WargameModelObservation,
)
from wargame_rl.wargame.model.dqn.device import Device, get_device


def observation_to_tensor(state: WargameEnvObservation, device: Device = None):
    device = get_device(device)
    norm = 25.0
    current_turn: int = state.current_turn
    wargame_models: list[WargameModelObservation] = state.wargame_models
    objectives: list[WargameEnvObjectiveObservation] = state.objectives
    tensor_current_turn = torch.tensor(
        [current_turn], dtype=torch.float32, device=device
    )
    tensor_wargame_models = torch.tensor(
        [model.location for model in wargame_models], dtype=torch.float32, device=device
    )
    tensor_objectives = torch.tensor(
        [objective.location for objective in objectives],
        dtype=torch.float32,
        device=device,
    )
    print(
        tensor_current_turn.shape, tensor_wargame_models.shape, tensor_objectives.shape
    )
    tensor_state = torch.cat(
        [
            tensor_current_turn,
            tensor_wargame_models.flatten(),
            tensor_objectives.flatten(),
        ],
        dim=0,
    )
    tensor_state = (tensor_state - norm) / norm
    return tensor_state.unsqueeze(0)


def observations_to_tensor_batch(
    states: list[WargameEnvObservation], device: Device = None
) -> torch.Tensor:
    device = get_device(device)

    return torch.cat([observation_to_tensor(state, device) for state in states], dim=0)
