# Define memory for Experience Replay
from collections import deque
from typing import Self

import torch

from wargame_rl.wargame.model.dqn.device import Device, get_device
from wargame_rl.wargame.model.dqn.state import state_to_tensor_batch
from wargame_rl.wargame.types import Experience, ExperienceBatch


def experience_list_to_batch(experiences: list[Experience]) -> ExperienceBatch:
    states_list = [experience.state for experience in experiences]
    next_states_list = [experience.new_state for experience in experiences]
    actions = [experience.action for experience in experiences]
    rewards = [experience.reward for experience in experiences]
    dones = [experience.done for experience in experiences]
    tensor_states = state_to_tensor_batch(states_list)
    tensor_next_states = state_to_tensor_batch(next_states_list)
    return ExperienceBatch(
        states=tensor_states,
        actions=torch.tensor(actions, dtype=torch.int32, device=tensor_states.device),
        rewards=torch.tensor(rewards, dtype=torch.float32, device=tensor_states.device),
        dones=torch.tensor(dones, dtype=torch.bool, device=tensor_states.device),
        new_states=tensor_next_states,
    )


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

    def sample(self, batch_size: int) -> ExperienceBatch:
        indices = torch.randint(0, len(self.buffer), (batch_size,))
        experiences = [self.buffer[idx] for idx in indices]
        return experience_list_to_batch(experiences)
