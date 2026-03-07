from typing import NamedTuple

import torch

from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvObservation


class Experience(NamedTuple):
    state: WargameEnvObservation
    action: WargameEnvAction
    reward: float
    done: bool
    new_state: WargameEnvObservation
    log_prob: torch.Tensor | None


class ExperienceBatch(NamedTuple):
    """Experience batch with typed fields."""

    state_tensors: list[torch.Tensor]
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    new_state_tensors: list[torch.Tensor]
    next_state_masks: torch.Tensor
    log_probs: torch.Tensor | None
