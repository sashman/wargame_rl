from __future__ import annotations

import numpy as np
from gymnasium import spaces


class WargameModel:
    """
    Wargame model class.

    Args:
        location: Location of the wargame model in the grid.

        stats: Statistics of the wargame model. Not used currently.

        distances_to_objectives: Distances to all objectives.

        group_id: Group ID of the wargame model. Should be a positive integer.
            The wargame models that belong to the same group will need to move in close proximity to each other.
            Otherwise they will receive a massive negative reward.
    """

    def __init__(
        self,
        location: np.ndarray,
        stats: dict[str, int],
        distances_to_objectives: np.ndarray,
        group_id: int,
    ):
        self.location = location
        self.stats = stats
        self.distances_to_objectives = distances_to_objectives
        self.group_id = group_id

    def __repr__(self) -> str:
        return f"WargameModel(location={self.location}, distances_to_objectives={self.distances_to_objectives}, group_id={self.group_id})"

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

        group_id_space = spaces.Discrete(1)

        return spaces.Dict(
            {
                "location": location_space,
                "distances_to_objectives": distances_to_objectives_space,
                "stats": stats_space,
                "group_id": group_id_space,
            }
        )
