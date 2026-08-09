"""Pure domain model for terrain footprints and LOS-blocking geometry."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Footprint:
    """Axis-aligned ruin footprint; stored normalised so x0<=x1 and y0<=y1.

    The bounds are **continuous board coordinates**, not cell indices: the piece
    covers the closed region ``[x0, x1] x [y0, y1]``. Configs and the random
    generator both author *cells*, so they come in through `from_cell_rect`,
    which is where the off-by-one lives.
    """

    x0: float
    y0: float
    x1: float
    y1: float

    def contains(self, x: float, y: float) -> bool:
        """True if the point (x, y) lies within the footprint (edge-inclusive)."""
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    @classmethod
    def from_corners(cls, x0: float, y0: float, x1: float, y1: float) -> "Footprint":
        """Create a Footprint from continuous corners, normalising the order."""
        return cls(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    @classmethod
    def from_cell_rect(cls, x0: int, y0: int, x1: int, y1: int) -> "Footprint":
        """Create a Footprint from an inclusive *cell* rectangle.

        A cell rect is corner-inclusive: ``(5, 5, 5, 5)`` names one cell, and
        ``(27, 8, 33, 16)`` names a 7x9 block. Read literally as a continuous
        rectangle those are zero area and 6x8 respectively, so the far corner is
        pushed out by one here. Nothing fails when this conversion is missed --
        the terrain just quietly gets smaller, which is why it lives at the
        single boundary rather than at each caller.
        """
        return cls(
            float(min(x0, x1)),
            float(min(y0, y1)),
            float(max(x0, x1)) + 1.0,
            float(max(y0, y1)) + 1.0,
        )


class Terrain:
    """Read-only collection of ruin footprints with the see-out endpoint filter."""

    def __init__(self, footprints: list[Footprint]) -> None:
        self._footprints = footprints

    @property
    def footprints(self) -> list[Footprint]:
        """All terrain footprints."""
        return self._footprints

    def blocking_footprints_for_endpoints(
        self, x0: float, y0: float, x1: float, y1: float
    ) -> list[Footprint]:
        """Footprints that can block this query: those containing NEITHER endpoint."""
        return [
            fp
            for fp in self._footprints
            if not fp.contains(x0, y0) and not fp.contains(x1, y1)
        ]
