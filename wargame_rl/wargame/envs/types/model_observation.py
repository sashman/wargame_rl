from dataclasses import dataclass

import numpy as np


@dataclass
class WargameModelObservation:
    """
    Observation structure for a Wargame model.
    """

    location: np.ndarray  # Location of the wargame model in the grid
    distances_to_objectives: np.ndarray  # Distances to all objectives
    group_id: int  # Group ID; models with the same group_id must stay within group_max_distance of at least one peer
    max_groups: int  # One-hot encoding size for group_id (from env config)
    alive: float  # 1.0 while the model has wounds remaining, else 0.0
    current_wounds: int
    max_wounds: int

    @property
    def size(self) -> int:
        """Location + distances + group one-hot + same-group distance + alive + wound scalars (3)."""
        return int(
            self.location.size
            + self.distances_to_objectives.size
            + self.max_groups
            + 1
            + 3
        )
