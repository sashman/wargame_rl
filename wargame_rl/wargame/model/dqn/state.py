import torch

from wargame_rl.wargame.model.dqn.device import Device, get_device
from wargame_rl.wargame.types import State


def state_to_tensor(state: State, device: Device = None) -> torch.Tensor:
    """Convert to tensor state and normalize"""

    agent = state["agent"]
    target = state["target"]
    flatten_state = [[agent[0], agent[1], target[0], target[1]]]
    norm = 25.0
    tensor_state = (
        torch.tensor(flatten_state, dtype=torch.float32, device=get_device(device))
        - norm
    ) / norm

    return tensor_state


def state_to_tensor_batch(states: list[State], device: Device = None) -> torch.Tensor:
    return torch.cat([state_to_tensor(state, device) for state in states], dim=0)
