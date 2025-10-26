from dataclasses import dataclass

import numpy as np


@dataclass
class WargameModelObservation:
    """
    Observation structure for a Wargame model.
    """

    location: np.ndarray  # Location of the wargame model in the grid
    distances_to_objectives: np.ndarray  # Distances to all objectives

    @property
    def size(self) -> int:
        return int(self.location.size + self.distances_to_objectives.size)
