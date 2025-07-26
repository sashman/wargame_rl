# Define memory for Experience Replay
from collections import deque
from typing import Self

import torch

from wargame_rl.wargame.model.dqn.device import Device, get_device
from wargame_rl.wargame.types import ExperienceV1


class ReplayBuffer:
    """Replay Buffer for storing past experiences allowing the agent to learn from them.

    Args:
        capacity: size of the buffer

    """

    def __init__(self, capacity: int, device: Device = None) -> None:
        self.buffer: deque[ExperienceV1] = deque(maxlen=capacity)
        self.device = get_device(device)

    def to(self, device: Device) -> Self:
        self.device = get_device(device)
        return self

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, experience: ExperienceV1) -> None:
        """Add experience to the buffer.

        Args:
            experience: tuple (state, action, reward, done, new_state)

        """
        self.buffer.append(experience)

    def sample_batch(self, batch_size: int) -> list[ExperienceV1]:
        # Sample without replacement using random permutation
        buffer_size = len(self.buffer)
        if batch_size > buffer_size:
            # If batch_size exceeds buffer size, return all experiences
            indices = torch.arange(buffer_size)
        else:
            # Sample without replacement
            indices = torch.randperm(buffer_size)[:batch_size]
        return [self.buffer[idx] for idx in indices]

    def sample(self) -> ExperienceV1:
        return self.buffer[torch.randint(0, len(self.buffer), (1,))]
