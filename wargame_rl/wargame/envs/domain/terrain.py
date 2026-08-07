"""Pure domain model for terrain footprints and LOS-blocking geometry."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wargame_rl.wargame.envs.types.geometry import Polygon


@dataclass(frozen=True, slots=True)
class Footprint:
    """The outline a terrain piece occupies, as a polygon.

    ``x0``/``y0``/``x1``/``y1`` remain available as the bounding box, because the
    exposure metric, the renderer and the observation all want a cheap extent. For a
    rectangular piece they are the piece itself.

    A footprint is the *outline*. Walls inside a ruin -- the L and U structures that
    break sight within a single piece -- are a separate feature that does not exist
    yet, so nothing here assumes the outline is convex.
    """

    polygon: Polygon
    _bounds: tuple[float, float, float, float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_bounds", self.polygon.bounds)

    @property
    def x0(self) -> float:
        return self._bounds[0]

    @property
    def y0(self) -> float:
        return self._bounds[1]

    @property
    def x1(self) -> float:
        return self._bounds[2]

    @property
    def y1(self) -> float:
        return self._bounds[3]

    @property
    def vertices(self) -> np.ndarray:
        """The outline, ``(n, 2)`` in board units."""
        return self.polygon.vertices

    def contains(self, x: float, y: float) -> bool:
        """True if point (x, y) lies within the footprint."""
        return self.polygon.contains(x, y)

    @classmethod
    def from_corners(cls, x0: float, y0: float, x1: float, y1: float) -> "Footprint":
        """Create a rectangular Footprint, normalising so x0<=x1 and y0<=y1."""
        return cls(Polygon.from_rect(x0, y0, x1, y1))

    @classmethod
    def from_cell_rect(cls, x0: int, y0: int, x1: int, y1: int) -> "Footprint":
        """Create a Footprint from a corner-inclusive rectangle of whole cells.

        Configs author terrain as the *cells* a piece covers, so ``(5, 5, 5, 5)`` is
        one cell and ``(27, 8, 33, 16)`` is seven by nine. Read literally as a
        continuous rectangle those would be zero-area and six by eight -- every piece
        a unit short on each axis. The far corner is pushed out by one so the area
        the config asked for is the area the piece has.
        """
        return cls(Polygon.from_cell_rect(x0, y0, x1, y1))

    @classmethod
    def from_points(cls, points: list[tuple[float, float]] | np.ndarray) -> "Footprint":
        """Create a Footprint from an explicit outline."""
        return cls(Polygon.from_points(points))


class Terrain:
    """Read-only collection of ruin footprints with the see-out endpoint filter."""

    def __init__(
        self,
        footprints: list[Footprint],
        blocking_mask: list[list[bool]] | None = None,
    ) -> None:
        self._footprints = footprints
        # Line of sight tests every blocker on every query, so the bounding boxes are
        # built once here. They are a cheap reject: only a piece whose box a ray
        # actually enters needs the exact polygon test.
        self._bounds = np.array(
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

        # Every outline padded to a common vertex count, so line of sight can test the
        # whole layout in one vectorised pass rather than one call per piece.
        budget = max((len(fp.vertices) for fp in footprints), default=3)
        self._padded_vertices = np.array(
            [fp.polygon.padded_vertices(budget) for fp in footprints], dtype=float
        ).reshape(-1, budget, 2)

    @property
    def footprints(self) -> list[Footprint]:
        """All terrain footprints. Excludes any legacy blocking mask."""
        return self._footprints

    @property
    def bounds(self) -> np.ndarray:
        """Footprint bounding boxes as an ``(n, 4)`` array of corners."""
        return self._bounds

    @property
    def padded_vertices(self) -> np.ndarray:
        """All outlines padded to a common vertex count, ``(n, v, 2)``."""
        return self._padded_vertices

    @property
    def mask_rectangles(self) -> np.ndarray:
        """Legacy blocking-mask cells as an ``(n, 4)`` array of corners."""
        return self._mask_rectangles

    def blocking_footprints_for_endpoints(
        self, x0: float, y0: float, x1: float, y1: float
    ) -> list[Footprint]:
        """Footprints that can block this query: those containing NEITHER endpoint.

        A model standing in a ruin can see out of it, and can be seen into it.
        """
        return [
            fp
            for fp in self._footprints
            if not fp.contains(x0, y0) and not fp.contains(x1, y1)
        ]

    def blocking_vertices_for_endpoints(
        self, x0: float, y0: float, x1: float, y1: float
    ) -> np.ndarray:
        """Padded outlines of the footprints that can block this query.

        Returns ``(n, max_vertices, 2)``, ready for the vectorised sampling test. The
        exact polygon test only runs for pieces whose bounding box actually holds an
        endpoint -- for the rectangular pieces that dominate most layouts the box test
        is already the answer, so the expensive check is rarely reached.
        """
        if not self._footprints:
            return self._padded_vertices
        holds = _box_holds(self._bounds, x0, y0) | _box_holds(self._bounds, x1, y1)
        for index in np.flatnonzero(holds):
            footprint = self._footprints[index]
            if not (footprint.contains(x0, y0) or footprint.contains(x1, y1)):
                holds[index] = False
        blocking: np.ndarray = self._padded_vertices[~holds]
        return blocking


def _box_holds(boxes: np.ndarray, x: float, y: float) -> np.ndarray:
    """``(n,)`` bool: which bounding boxes contain the point."""
    held: np.ndarray = (
        (boxes[:, 0] <= x)
        & (x <= boxes[:, 2])
        & (boxes[:, 1] <= y)
        & (y <= boxes[:, 3])
    )
    return held
