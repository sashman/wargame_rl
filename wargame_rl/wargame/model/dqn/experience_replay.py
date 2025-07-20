# Define memory for Experience Replay
from collections import deque
from typing import Self, Tuple, NamedTuple
import torch


from wargame_rl.wargame.model.dqn.device import Device, get_device


class Experience(NamedTuple):
    """Experience tuple with typed fields."""

    state: torch.Tensor
    action: int
    reward: float
    done: bool
    new_state: torch.Tensor


class ReplayBuffer:
    """Replay Buffer for storing past experiences allowing the agent to learn from them.

    Args:
        capacity: size of the buffer

    """

    def __init__(self, capacity: int, device: Device = None) -> None:
        self.buffer = deque(maxlen=capacity)
        self.device = get_device(device)

    def to(self, device: Device) -> Self:
        self.device = get_device(device)
        return self

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """Add experience to the buffer.

        Args:
            experience: tuple (state, action, reward, done, new_state)

        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        indices = torch.randint(0, len(self.buffer), (batch_size,))
        states, actions, rewards, dones, next_states = zip(
            *(self.buffer[idx] for idx in indices)
        )

        return (
            torch.tensor(states, device=self.device),
            torch.tensor(actions, device=self.device),
            torch.tensor(rewards, dtype=torch.float32, device=self.device),
            torch.tensor(dones, dtype=bool, device=self.device),
            torch.tensor(next_states, device=self.device),
        )
