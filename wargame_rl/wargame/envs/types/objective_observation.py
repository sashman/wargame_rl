from dataclasses import dataclass

import numpy as np


@dataclass
class WargameEnvObjectiveObservation:
    """
    Observation structure for a Wargame objective.
    """

    location: np.ndarray  # Location of the objective in the grid

    @property
    def size(self) -> int:
        return self.location.size
