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

    # The vector from this model to its unit's live centroid, over the spread
    # cap, clipped per axis. Populated only when `observe_unit_centroid` is set.
    #
    # The two scalars above are **magnitudes** — they say a unit is stretched or
    # split, not which way to move. The spread ratio reads the same from either
    # side of a strung-out unit, so a model cannot act on it. This is the
    # direction, and it is the quantity the scripted demonstrators actually
    # compute: `squad_march` steers a whole unit along one shared centroid
    # vector, which is *why* their formation holds by construction.
    #
    # Clipped per axis rather than by magnitude so the sign survives however far
    # a model has strayed. Losing "how far" costs nothing here — that is exactly
    # what `coherency_spread` carries.
    unit_offset: np.ndarray | None = None
    # The two halves of the advance trade, present only when the scenario has
    # advance bins. Without them a policy choosing an advance is blind to both
    # what it buys and what it has already spent:
    #   `advance_roll`      -- this model's UNIT's D6 for the turn, normalised
    #                          by the die's maximum. The rules roll before the
    #                          move is chosen, so it must be observable when the
    #                          choice is made, not merely afterwards.
    #   `advanced_this_turn` -- whether it has already advanced. The action mask
    #                          already FORBIDS shooting, so the policy cannot act
    #                          on it wrongly; this is for the VALUE head, which
    #                          otherwise cannot tell a model that has spent its
    #                          shooting from one holding fire by choice.
    advance_roll: float | None = None
    advanced_this_turn: float | None = None
    # The two halves of the melee trade, present only when the scenario fights
    # in melee. They are the charge's exact analogues of the advance pair above,
    # and they exist for the same reason: without them the trunk cannot condition
    # on a roll that the action mask has already spent, and the value head cannot
    # tell a model that has forfeited its shooting from one holding fire.
    #   `charge_roll`         -- this model's UNIT's 2D6 for the turn, normalised
    #                            by the maximum the two dice can show. The rules
    #                            roll before the charge is declared, so it must be
    #                            observable when the choice is made.
    #   `fell_back_this_turn` -- whether its unit fell back out of melee, which
    #                            costs it both its shooting and its charge.
    #
    # ⚠ `charged_this_turn` is deliberately NOT here. It is cleared the moment
    # the fight it governs resolves, which happens inside the same step, so an
    # observation could only ever read it as False -- a constant column.
    charge_roll: float | None = None
    fell_back_this_turn: float | None = None

    @property
    def size(self) -> int:
        """Location + distances + group one-hot + same-group distance + alive + wound scalars (3) + combat stats (7), plus unit strength, objective-presence flags, the two coherency scalars, the two advance scalars and the two melee scalars when observed."""
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
            + (0 if self.unit_offset is None else self.unit_offset.size)
            + (0 if self.advance_roll is None else 1)
            + (0 if self.advanced_this_turn is None else 1)
            + (0 if self.charge_roll is None else 1)
            + (0 if self.fell_back_this_turn is None else 1)
        )
