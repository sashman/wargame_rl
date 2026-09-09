"""The activation: which unit is open in a phase, and who in it may act next.

`docs/rules/03-moving.md` and the phase chapters share one sequencing rule --
*select one friendly unit that has not been selected this phase, resolve it
model by model, then select the next* -- and this module is that rule written
once. A `PhaseActivation` holds one seat's state for one phase: who has acted,
which unit is open, who opened it, and the unit-level choices made on the way
in. It references models by index and holds no model objects, so it is a
phase-transient record beside the `Battle` aggregate rather than a rival to it.

The declarations are the unit-level choices the rules make on a unit's opening
step: the move type in movement, whether a unit shoots at all, whether it
charges, whether it piles in or consolidates. Each is a small value object so a
phase can say which values are legal and a step can name one.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class MoveDeclaration(IntEnum):
    """A unit's move type for the movement phase (`09-movement-phase.md`)."""

    stationary = 0
    normal = 1
    advance = 2
    fall_back = 3


class ShootDeclaration(IntEnum):
    """Whether a unit shoots this phase. Exists so a silent unit costs one step."""

    hold_fire = 0
    shoot = 1


class ChargeDeclaration(IntEnum):
    """Whether a unit declares a charge (`11-charge-phase.md` step 1)."""

    decline = 0
    charge = 1


class ShortMoveDeclaration(IntEnum):
    """Whether an eligible unit piles in / consolidates (`12-fight-phase.md`)."""

    decline = 0
    move = 1


class ConsolidationMode(IntEnum):
    """The compulsory consolidation mode, assessed in this order."""

    none = 0
    ongoing = 1
    engaging = 2
    objective = 3


# The charge-target step's way of saying "now that the roll is known, do not
# charge after all" -- `11-charge-phase.md` step 3 grants exactly that.
CHARGE_TARGET_DECLINE = -1


class ActivationError(ValueError):
    """An activation invariant was violated by the caller."""


class PhaseActivation:
    """One seat's unit lock for one phase.

    Invariants, checked on every mutation: at most one unit is open; a model
    acts only while its unit is open; a unit that has closed never reopens in
    the phase; a forced model is the only selectable one while it is set.
    """

    def __init__(self, n_models: int) -> None:
        self.acted = np.zeros(n_models, dtype=bool)
        self.closed_units: set[int] = set()
        self.open_unit: int | None = None
        self.opener: int | None = None
        self.forced_model: int | None = None
        self.declaration: int | None = None
        self.charge_target: int | None = None
        self.start_positions: dict[int, np.ndarray] = {}

    @property
    def is_open(self) -> bool:
        """True while a unit is activated and not yet closed."""
        return self.open_unit is not None

    def selectable(
        self,
        alive: np.ndarray,
        group_ids: np.ndarray,
        units_in_phase: set[int],
    ) -> np.ndarray:
        """Which models may be named next, `(n_models,)` bool.

        Alive and not yet acted always; then the forced model alone if one is
        set, else the open unit's members, else any member of a unit that is in
        this phase and has not closed.
        """
        mask = alive & ~self.acted
        if self.forced_model is not None:
            forced: np.ndarray = np.zeros_like(mask)
            forced[self.forced_model] = mask[self.forced_model]
            return forced
        if self.open_unit is not None:
            in_unit: np.ndarray = mask & (group_ids == self.open_unit)
            return in_unit
        allowed = np.array(
            [
                int(group) in units_in_phase and int(group) not in self.closed_units
                for group in group_ids
            ],
            dtype=bool,
        )
        selectable: np.ndarray = mask & allowed
        return selectable

    def open(
        self,
        unit: int,
        opener: int,
        declaration: int | None,
        *,
        force_opener: bool,
    ) -> None:
        """Activate `unit` through `opener`; optionally bind the next step to it."""
        if self.open_unit is not None:
            raise ActivationError(f"unit {self.open_unit} is already open")
        if unit in self.closed_units:
            raise ActivationError(f"unit {unit} has already closed this phase")
        self.open_unit = unit
        self.opener = opener
        self.declaration = declaration
        self.forced_model = opener if force_opener else None

    def set_charge_target(self, target: int) -> None:
        """Record the target chosen on the charge's second unit-level step."""
        if self.open_unit is None:
            raise ActivationError("no unit is open to take a charge target")
        self.charge_target = target

    def mark_acted(self, model: int, group: int) -> None:
        """One model of the open unit has taken its action."""
        if self.open_unit is None or group != self.open_unit:
            raise ActivationError(f"model {model} acted outside its open unit")
        if self.acted[model]:
            raise ActivationError(f"model {model} has already acted this phase")
        self.acted[model] = True
        self.forced_model = None

    def force(self, model: int) -> None:
        """Bind the next selection to one model of the open unit."""
        if self.open_unit is None:
            raise ActivationError("no unit is open to force a model in")
        self.forced_model = model

    def close(self) -> None:
        """Close the open unit; it cannot reopen this phase."""
        if self.open_unit is None:
            raise ActivationError("no unit is open to close")
        self.closed_units.add(self.open_unit)
        self.open_unit = None
        self.opener = None
        self.forced_model = None
        self.declaration = None
        self.charge_target = None
        self.start_positions = {}

    def close_without_steps(self, unit: int, members: list[int]) -> None:
        """Close a unit that never opened, marking its members acted.

        For a unit whose only legal declaration is the closing one -- a legality
        fact the env states at phase open rather than a step it makes the policy
        take.
        """
        if self.open_unit == unit:
            raise ActivationError(f"unit {unit} is open; close it through close()")
        for index in members:
            self.acted[index] = True
        self.closed_units.add(unit)
