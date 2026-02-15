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
    def to_space(board_width: int, board_height: int) -> spaces.Dict:
        return spaces.Dict(
            {
                "location": spaces.Box(
                    low=np.array([0, 0], dtype=np.int32),
                    high=np.array([board_width - 1, board_height - 1], dtype=np.int32),
                    shape=(2,),
                    dtype=np.int32,
                )
            }
        )
