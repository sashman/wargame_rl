from __future__ import annotations

import numpy as np
from gymnasium import spaces


class WargameObjective:
    def __init__(self, location: np.ndarray, radius_size: int):
        self.location = location  # Should be a numpy array of shape (2,)
        self.radius_size = radius_size  # Radius of the objective in the environment

    def __repr__(self) -> str:
        return f"WargameObjective(location={self.location}, radius_size={self.radius_size})"

    @staticmethod
    def to_space(size: int) -> spaces.Dict:
        return spaces.Dict(
            {"location": spaces.Box(0, size - 1, shape=(2,), dtype=np.int32)}
        )
