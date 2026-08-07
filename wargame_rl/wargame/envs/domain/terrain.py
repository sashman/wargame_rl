"""Pure domain model for terrain footprints and LOS-blocking geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Footprint:
    """Axis-aligned ruin footprint; stored normalised so x0<=x1 and y0<=y1."""

    x0: float
    y0: float
    x1: float
    y1: float

    def contains(self, x: float, y: float) -> bool:
        """True if point (x, y) lies within the footprint (corner-inclusive)."""
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    @classmethod
    def from_corners(cls, x0: float, y0: float, x1: float, y1: float) -> "Footprint":
        """Create a Footprint, normalising so x0<=x1 and y0<=y1."""
        return cls(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    @classmethod
    def from_cell_rect(cls, x0: int, y0: int, x1: int, y1: int) -> "Footprint":
        """Create a Footprint from a corner-inclusive rectangle of whole cells.

        Configs author terrain as the *cells* a piece covers, so ``(5, 5, 5, 5)`` is
        one cell and ``(27, 8, 33, 16)`` is seven by nine. Read literally as a
        continuous rectangle those would be zero-area and six by eight -- every piece
        a unit short on each axis. The far corner is pushed out by one so the area
        the config asked for is the area the piece has.
        """
        lo_x, hi_x = min(x0, x1), max(x0, x1)
        lo_y, hi_y = min(y0, y1), max(y0, y1)
        return cls(float(lo_x), float(lo_y), float(hi_x + 1), float(hi_y + 1))


class Terrain:
    """Read-only collection of ruin footprints with the see-out endpoint filter."""

    def __init__(
        self,
        footprints: list[Footprint],
        blocking_mask: list[list[bool]] | None = None,
    ) -> None:
        self._footprints = footprints
        # Line of sight is the hottest path in the environment and tests every
        # blocker on every query, so the rectangle arrays are built once here rather
        # than rebuilt per query.
        self._footprint_rectangles = np.array(
            [[fp.x0, fp.y0, fp.x1, fp.y1] for fp in footprints], dtype=float
        ).reshape(-1, 4)

        # The legacy board-sized mask is one unit square per set cell. It is kept
        # separate from the footprints for two reasons: those become tokens in the
        # observation, and a mask would flood it with hundreds of one-unit pieces;
        # and a mask square is opaque matter rather than a feature a model can stand
        # inside, so the see-out rule below does not apply to it.
        self._mask_rectangles = np.array(
            [
                [float(x), float(y), float(x + 1), float(y + 1)]
                for y, row in enumerate(blocking_mask or [])
                for x, blocked in enumerate(row)
                if blocked
            ],
            dtype=float,
        ).reshape(-1, 4)

    @property
    def footprints(self) -> list[Footprint]:
        """All terrain footprints. Excludes any legacy blocking mask."""
        return self._footprints

    @property
    def rectangles(self) -> np.ndarray:
        """Everything that blocks sight, as an ``(n, 4)`` array of corners."""
        return np.vstack([self._footprint_rectangles, self._mask_rectangles])

    def blocking_rectangles_for_endpoints(
        self, x0: float, y0: float, x1: float, y1: float
    ) -> np.ndarray:
        """Everything that can block this query, as an ``(n, 4)`` array.

        A *footprint* containing either endpoint is dropped: a model standing in a
        ruin can see out of it, and can be seen into it. Mask squares are never
        dropped -- they are solid, not sheltering.
        """
        rects = self._footprint_rectangles
        if rects.size == 0:
            return self._mask_rectangles
        holds_start = (
            (rects[:, 0] <= x0)
            & (x0 <= rects[:, 2])
            & (rects[:, 1] <= y0)
            & (y0 <= rects[:, 3])
        )
        holds_end = (
            (rects[:, 0] <= x1)
            & (x1 <= rects[:, 2])
            & (rects[:, 1] <= y1)
            & (y1 <= rects[:, 3])
        )
        return np.vstack([rects[~(holds_start | holds_end)], self._mask_rectangles])

    def blocking_footprints_for_endpoints(
        self, x0: float, y0: float, x1: float, y1: float
    ) -> list[Footprint]:
        """Footprints that can block this query: those containing NEITHER endpoint."""
        return [
            fp
            for fp in self._footprints
            if not fp.contains(x0, y0) and not fp.contains(x1, y1)
        ]
