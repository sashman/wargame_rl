from dataclasses import dataclass

import numpy as np


@dataclass
class WargameEnvObjectiveObservation:
    """
    Observation structure for a Wargame objective.
    """

    location: np.ndarray  # Location of the objective in the grid

    # Control state, populated only when `observe_objective_control` is set.
    # None keeps the token at two dimensions, which is what every existing
    # checkpoint expects.
    #
    # These exist because VP is scored on `player_count > opponent_count` per
    # objective, and until now an objective reached the network as *nothing but
    # a location*. The agent was asked to optimise a strict count comparison
    # over two unordered 25-token sets, with no positional encoding and no
    # counting primitive -- and any reward keyed on those counts (the
    # `objective_hold` surplus discount, the `closest_objective_v2` overstack
    # penalty) was therefore unattributable: the policy could only experience it
    # as "standing on objectives pays less", and did less of it.
    player_count: float | None = None
    opponent_count: float | None = None
    radius: float | None = None

    @property
    def size(self) -> int:
        return int(self.location.size)
