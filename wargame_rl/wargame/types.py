from typing import NamedTuple, TypedDict

import numpy as np
import torch


class State(TypedDict):
    agent: np.ndarray
    target: np.ndarray


class Experience(NamedTuple):
    """Experience tuple with typed fields."""

    state: State
    action: int
    reward: float
    done: bool
    new_state: State


class ExperienceBatch(NamedTuple):
    """Experience batch with typed fields."""

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    new_states: torch.Tensor
