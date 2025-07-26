import os
from typing import Iterator

import torch
from torch.utils.data.dataset import IterableDataset

from wargame_rl.wargame.model.dqn.experience_replay import ReplayBuffer
from wargame_rl.wargame.model.dqn.state import state_to_tensor_v1_batch
from wargame_rl.wargame.types import ExperienceBatch, ExperienceV1

PATH_DATASETS = os.environ.get("PATH_DATASETS", "./datasets")


def experience_list_to_batch(experiences: list[ExperienceV1]) -> ExperienceBatch:
    states_list = [experience.state for experience in experiences]
    next_states_list = [experience.new_state for experience in experiences]
    actions = [experience.action for experience in experiences]
    rewards = [experience.reward for experience in experiences]
    dones = [experience.done for experience in experiences]
    tensor_states = state_to_tensor_v1_batch(states_list)
    tensor_next_states = state_to_tensor_v1_batch(next_states_list)

    return ExperienceBatch(
        states=tensor_states,
        actions=torch.tensor(actions, dtype=torch.int32, device=tensor_states.device),
        rewards=torch.tensor(rewards, dtype=torch.float32, device=tensor_states.device),
        dones=torch.tensor(dones, dtype=torch.bool, device=tensor_states.device),
        new_states=tensor_next_states,
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

    def __iter__(self) -> Iterator[ExperienceV1]:
        if len(self.buffer) > 0:
            while True:
                yield self.buffer.sample()

    def __len__(self) -> int:
        return self.sample_size
