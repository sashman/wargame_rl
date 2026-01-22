import os
from typing import Iterator

import torch
from torch.utils.data.dataset import IterableDataset

from wargame_rl.wargame.model.dqn.experience_replay import ReplayBuffer
from wargame_rl.wargame.model.dqn.observation import observations_to_tensor_batch
from wargame_rl.wargame.types import Experience, ExperienceBatch

PATH_DATASETS = os.environ.get("PATH_DATASETS", "./datasets")


def experience_list_to_batch(experiences: list[Experience]) -> ExperienceBatch:
    states_list = [experience.state for experience in experiences]
    next_states_list = [experience.new_state for experience in experiences]
    actions = [experience.action.actions for experience in experiences]
    rewards = [experience.reward for experience in experiences]
    dones = [experience.done for experience in experiences]
    tensor_states: list[torch.Tensor] = observations_to_tensor_batch(states_list)
    tensor_next_states: list[torch.Tensor] = observations_to_tensor_batch(
        next_states_list
    )
    device = tensor_states[0].device

    return ExperienceBatch(
        state_tensors=tensor_states,
        actions=torch.tensor(actions, dtype=torch.int32, device=device),
        rewards=torch.tensor(rewards, dtype=torch.float32, device=device),
        dones=torch.tensor(dones, dtype=torch.bool, device=device),
        new_state_tensors=tensor_next_states,
    )


class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated with new experiences during training.

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time

    """

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 1024) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Iterator[Experience]:
        if len(self.buffer) > 0:
            while True:
                yield self.buffer.sample()

    def __len__(self) -> int:
        return self.sample_size
