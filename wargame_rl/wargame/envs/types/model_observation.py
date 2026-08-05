from dataclasses import dataclass

import numpy as np


@dataclass
class WargameModelObservation:
    """Observation structure for a Wargame model."""

    location: np.ndarray
    distances_to_objectives: np.ndarray
    group_id: int
    max_groups: int
    alive: float
    current_wounds: int
    max_wounds: int
    weapon_attacks: int = 0
    weapon_ballistic_skill: int = 0
    weapon_strength: int = 0
    weapon_ap: int = 0
    weapon_damage: int = 0
    toughness: int = 0
    save_stat: int = 0
    # Fraction of the opposing force with line of sight and weapon range to
    # this model. None when `observe_threat_count` is off, which is also what
    # keeps the observation width unchanged for existing configs.
    threat_count: float | None = None

    @property
    def size(self) -> int:
        """Location + distances + group one-hot + same-group distance + optional threat + alive + wound scalars (3) + combat stats (7)."""
        return int(
            self.location.size
            + self.distances_to_objectives.size
            + self.max_groups
            + 1
            + (0 if self.threat_count is None else 1)
            + 3
            + 7
        )
