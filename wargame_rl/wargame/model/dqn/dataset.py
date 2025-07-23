from typing import Iterator, Tuple
from torch.utils.data.dataset import IterableDataset
from wargame_rl.wargame.model.dqn.experience_replay import ReplayBuffer
import os
from wargame_rl.wargame.types import ExperienceBatch

PATH_DATASETS = os.environ.get("PATH_DATASETS", "./datasets")


class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated with new experiences during training.

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time

    """

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 128) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Iterator[ExperienceBatch]:
        return iter(self.buffer.sample(self.sample_size))
