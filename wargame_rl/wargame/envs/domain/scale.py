"""The mapping between rules inches and environment coordinate units.

The rules are written in inches. The environment plays on a board whose unit is
arbitrary, so every rules distance has to be converted before it can be compared
against a coordinate. ``Scale`` is that conversion, and it is the only place the
relationship is defined.

The convention across the codebase:

* **Coordinates, and anything that produces one** -- board dimensions, model and
  objective positions, terrain footprints, deployment zones -- are in **units**.
* **Rules distances** -- move allowance, weapon range, engagement range -- are
  authored in **inches** and converted here.

At the default of one inch per unit the two coincide and conversion is the
identity, which is what lets the scale be introduced without changing a single
measured result.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Scale:
    """How many rules inches one environment coordinate unit spans."""

    inches_per_unit: float = 1.0

    def __post_init__(self) -> None:
        if not self.inches_per_unit > 0:
            raise ValueError(
                f"inches_per_unit must be positive, got {self.inches_per_unit}"
            )

    def to_units(self, inches: float) -> float:
        """Convert a rules distance in inches to environment coordinate units."""
        return inches / self.inches_per_unit

    def to_inches(self, units: float) -> float:
        """Convert an environment distance in coordinate units to rules inches."""
        return units * self.inches_per_unit
