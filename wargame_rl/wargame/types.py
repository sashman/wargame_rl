from typing import NamedTuple

import torch

from wargame_rl.wargame.envs.env_types import WargameEnvObservation


class Experience(NamedTuple):
    state: WargameEnvObservation
    action: int
    reward: float
    done: bool
    new_state: WargameEnvObservation


class ExperienceBatch(NamedTuple):
    """Experience batch with typed fields."""

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    new_states: torch.Tensor
