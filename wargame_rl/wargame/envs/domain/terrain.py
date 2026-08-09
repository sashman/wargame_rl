"""Pure domain model for terrain footprints and LOS-blocking geometry."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Footprint:
    """Axis-aligned ruin footprint; stored normalised so x0<=x1 and y0<=y1."""

    x0: int
    y0: int
    x1: int
    y1: int

    def contains(self, x: int, y: int) -> bool:
        """True if cell (x, y) lies within the footprint (corner-inclusive)."""
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    @classmethod
    def from_corners(cls, x0: int, y0: int, x1: int, y1: int) -> "Footprint":
        """Create a Footprint, normalising so x0<=x1 and y0<=y1."""
        return cls(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


class Terrain:
    """Read-only collection of ruin footprints with the see-out endpoint filter."""

    def __init__(self, footprints: list[Footprint]) -> None:
        self._footprints = footprints

    @property
    def footprints(self) -> list[Footprint]:
        """All terrain footprints."""
        return self._footprints

    def blocking_footprints_for_endpoints(
        self, x0: int, y0: int, x1: int, y1: int
    ) -> list[Footprint]:
        """Footprints that can block this query: those containing NEITHER endpoint."""
        return [
            fp
            for fp in self._footprints
            if not fp.contains(x0, y0) and not fp.contains(x1, y1)
        ]
