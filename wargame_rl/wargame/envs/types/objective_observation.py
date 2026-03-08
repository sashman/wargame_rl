from dataclasses import dataclass

import numpy as np


@dataclass
class WargameEnvObjectiveObservation:
    """
    Observation structure for a Wargame objective.
    """

    location: np.ndarray  # Location of the objective in the grid
    player_level_of_control: float = 0.0  # Sum of OC of player models in range
    opponent_level_of_control: float = 0.0  # Sum of OC of opponent models in range

    @property
    def size(self) -> int:
        return (
            int(self.location.size) + 2
        )  # location (2) + player_loc (1) + opponent_loc (1)
