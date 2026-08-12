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

    # 1.0 for a real objective, 0.0 for a padding slot; populated only when
    # `objective_budget` is set. None keeps the token at its historical width.
    #
    # Padding is otherwise indistinguishable from a real objective at the board
    # centre with nobody on it -- the same reason a terrain token carries its
    # vertex count. It is also what makes "the row is entirely zero" a safe test
    # for padding, which is how the network drops these from attention.
    present: float | None = None

    @property
    def size(self) -> int:
        return int(self.location.size)
