from __future__ import annotations

import numpy as np
from gymnasium import spaces


class WargameObjective:
    def __init__(self, location: np.ndarray, radius_size: int):
        self.location = location  # Should be a numpy array of shape (2,)
        self.radius_size = radius_size  # Radius of the objective in the environment

    def __repr__(self) -> str:
        return f"WargameObjective(location={self.location}, radius_size={self.radius_size})"

    # Max value for radius_size in the observation space. Discrete(n) has n elements
    # (0 to n-1), so we use MAX + 1 to allow 0..MAX.
    MAX_RADIUS_FOR_SPACE = 100
    # Max distance for closest_*_distance observation fields (board diagonal scale).
    MAX_DISTANCE_FOR_SPACE = 1500

    @staticmethod
    def to_space(board_width: int, board_height: int) -> spaces.Dict:
        max_d = float(WargameObjective.MAX_DISTANCE_FOR_SPACE)
        return spaces.Dict(
            {
                "location": spaces.Box(
                    low=np.array([0, 0], dtype=np.int32),
                    high=np.array([board_width - 1, board_height - 1], dtype=np.int32),
                    shape=(2,),
                    dtype=np.int32,
                ),
                "radius_size": spaces.Discrete(
                    WargameObjective.MAX_RADIUS_FOR_SPACE + 1
                ),  # single scalar: 0..MAX_RADIUS_FOR_SPACE
                "closest_player_distance": spaces.Box(
                    low=0.0, high=max_d, shape=(), dtype=np.float32
                ),
                "closest_opponent_distance": spaces.Box(
                    low=0.0, high=max_d, shape=(), dtype=np.float32
                ),
            }
        )
