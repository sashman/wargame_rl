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

    @property
    def size(self) -> int:
        """Location + distances_to_objectives + one-hot group_id (length max_groups)."""
        return int(
            self.location.size + self.distances_to_objectives.size + self.max_groups
        )
