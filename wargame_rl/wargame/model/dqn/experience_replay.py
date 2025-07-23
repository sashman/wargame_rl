# Define memory for Experience Replay
from collections import deque
from typing import Self

import torch

from wargame_rl.wargame.model.dqn.device import Device, get_device
from wargame_rl.wargame.types import Experience


class ReplayBuffer:
    """Replay Buffer for storing past experiences allowing the agent to learn from them.

    Args:
        capacity: size of the buffer

    """

    def __init__(self, capacity: int, device: Device = None) -> None:
        self.buffer: deque[Experience] = deque(maxlen=capacity)
        self.device = get_device(device)

    def to(self, device: Device) -> Self:
        self.device = get_device(device)
        return self

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """Add experience to the buffer.

        Args:
            experience: tuple (state, action, reward, done, new_state)

        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> list[Experience]:
        indices = torch.randint(0, len(self.buffer), (batch_size,))
        return [self.buffer[idx] for idx in indices]
