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
    radius_size: int = 0  # Radius of the objective (for capture / control range)
    closest_player_distance: float = (
        0.0  # Min distance from any player model to this objective
    )
    closest_opponent_distance: float = (
        0.0  # Min distance from any opponent to this objective (sentinel when none)
    )

    @property
    def size(self) -> int:
        # location (2) + player_loc (1) + opponent_loc (1) + radius_size (1) + closest_* (2)
        return int(self.location.size) + 5
