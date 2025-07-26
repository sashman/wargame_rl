from typing import NamedTuple, TypedDict

import numpy as np
import torch

from wargame_rl.wargame.envs.env_types import WargameEnvObservation


class StateV1(TypedDict):
    agent: np.ndarray
    target: np.ndarray


class ExperienceV1(NamedTuple):
    """Experience tuple with typed fields."""

    state: StateV1
    action: int
    reward: float
    done: bool
    new_state: StateV1


class ExperienceV2(NamedTuple):
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
