from __future__ import annotations

import numpy as np
from gymnasium import spaces


class WargameModel:
    def __init__(
        self, location: np.ndarray, stats: dict, distances_to_objectives: np.ndarray
    ):
        self.location = location  # Should be a numpy array of shape (2,)
        self.stats = (
            stats  # Should be a dictionary with keys 'max_wounds' and 'current_wounds'
        )
        self.distances_to_objectives = distances_to_objectives  # Should be a numpy array of shape (number_of_objectives, 2)

    def __repr__(self) -> str:
        return f"WargameModel(location={self.location}, distances_to_objectives={self.distances_to_objectives}, stats={self.stats})"

    @staticmethod
    def to_space(size: int, number_of_objectives: int) -> spaces.Dict:
        location_space = spaces.Box(0, size - 1, shape=(2,), dtype=np.int32)
        distances_to_objectives_space = spaces.Box(
            0, size - 1, shape=(number_of_objectives, 2), dtype=np.int32
        )
        stats_space = spaces.Dict(
            {
                "max_wounds": spaces.Box(0, 100, shape=(1,), dtype=np.int32),
                "current_wounds": spaces.Box(0, 100, shape=(1,), dtype=np.int32),
            }
        )

        return spaces.Dict(
            {
                "location": location_space,
                "distances_to_objectives": distances_to_objectives_space,
                "stats": stats_space,
            }
        )
