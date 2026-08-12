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

    # Fraction of this model's *unit* still alive, populated only when
    # `observe_unit_strength` is set. None keeps the token at its historical
    # width, which is what every existing checkpoint expects.
    #
    # Shooting names a unit and the defender allocates, so how many models a
    # unit has left decides whether a volley finishes it or is thrown at a full
    # one. That count was in no input: the shooting head mean-pools opponent
    # tokens into one per unit, and a mean is invariant to how many terms it
    # averages.
    #
    # It is deliberately a per-model column carrying a per-unit quantity, so it
    # is identical across a unit's members and every member's token states it.
    # The pooling averages post-transformer *latents*, not these features, so
    # this does not make the count survive pooling by arithmetic -- the claim is
    # only the weaker and sufficient one, that the quantity is now present in
    # the input at all, on tokens the head and the trunk both read.
    unit_strength: float | None = None

    # One flag per objective slot, 1.0 real and 0.0 padding, populated only when
    # `objective_budget` is set. `distances_to_objectives` is padded to the same
    # budget, and a padding slot's delta is (0, 0) -- which without this column
    # reads as "this model is standing on that objective", the most emphatic
    # thing the feature can say. The flags are identical across every model on
    # the board, but the distance columns they qualify are per model, so this is
    # where they have to live.
    objective_present: np.ndarray | None = None

    # The two halves of the coherency rule that no input carried, populated only
    # when `observe_coherency` is set.
    #
    # `coherency_spread` is the distance to the *furthest* live model in this
    # model's unit, over the spread cap -- the 9" condition, which had no tensor
    # at all. The existing same-group column is a *nearest* neighbour distance,
    # so a unit strung across the board reads as perfectly tight from every
    # model in it as long as each has a partner.
    #
    # `coherency_component` is the fraction of the unit's live models in this
    # model's own chain component. It is 1.0 for a unit in one piece and drops
    # when the unit splits, which is the rule's third clause -- and the one that
    # is a transitive closure, so no amount of attention over pairwise distances
    # recovers it cheaply.
    #
    # Both are normalised by the coherency distances rather than by the board
    # diagonal. That is the point: against the diagonal the whole 2" band is
    # 2.7% of a column's range, so the decision-relevant region is compressed
    # into noise.
    coherency_spread: float | None = None
    coherency_component: float | None = None

    @property
    def size(self) -> int:
        """Location + distances + group one-hot + same-group distance + alive + wound scalars (3) + combat stats (7), plus unit strength, objective-presence flags and the two coherency scalars when observed."""
        return int(
            self.location.size
            + self.distances_to_objectives.size
            + self.max_groups
            + 1
            + 3
            + 7
            + (0 if self.unit_strength is None else 1)
            + (0 if self.objective_present is None else self.objective_present.size)
            + (0 if self.coherency_spread is None else 1)
            + (0 if self.coherency_component is None else 1)
        )
