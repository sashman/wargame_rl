"""Terrain footprints and the LOS-blocking geometry they carry.

A footprint is a `Polygon` — an outline, not a bounding box. The distinction is
the point: until this, an L-shaped ruin and a solid block reached both the sight
trace and the network as the same four numbers, so "can the agent use terrain?"
was being asked of an input that could not express the question.

Rectangles are still the common case and are still authored as *inclusive cell*
rectangles, which is why `from_cell_rect` exists here as well as on `Polygon`.
"""

from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs.types.geometry import Polygon


class Footprint:
    """One terrain piece: a closed outline in continuous board units.

    `x0` / `y0` / `x1` / `y1` remain available as the outline's bounding box.
    They are a *summary*, not the shape — anything deciding whether a point is
    in the piece must go through `contains`, or a concave ruin's notch counts as
    solid.
    """

    __slots__ = ("_polygon", "_bounds")

    def __init__(self, polygon: Polygon) -> None:
        self._polygon = polygon
        self._bounds = polygon.bounds

    @property
    def polygon(self) -> Polygon:
        """The outline itself."""
        return self._polygon

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
    def n_vertices(self) -> int:
        """Number of vertices in the outline."""
        return self._polygon.n_vertices

    def contains(self, x: float, y: float) -> bool:
        """True if the point lies within the outline (edge-inclusive)."""
        return self._polygon.contains(x, y)

    def distance_to_point(self, x: float, y: float) -> float:
        """Distance to the outline; 0.0 anywhere inside it."""
        return self._polygon.distance_to_point(x, y)

    def overlaps(self, other: "Footprint") -> bool:
        """True if the two pieces share interior area. Touching does not count."""
        return self._polygon.overlaps(other._polygon)

    def __repr__(self) -> str:
        return f"Footprint(bounds={self._bounds}, n_vertices={self.n_vertices})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Footprint):
            return NotImplemented
        return bool(np.array_equal(self._polygon.vertices, other._polygon.vertices))

    @classmethod
    def from_corners(cls, x0: float, y0: float, x1: float, y1: float) -> "Footprint":
        """Build an axis-aligned piece from continuous corners."""
        return cls(Polygon.from_rect(x0, y0, x1, y1))

    @classmethod
    def from_cell_rect(cls, x0: int, y0: int, x1: int, y1: int) -> "Footprint":
        """Build from an inclusive *cell* rectangle, pushing the far corner out by one.

        A cell rect is corner-inclusive: ``(5, 5, 5, 5)`` names one cell, and
        ``(27, 8, 33, 16)`` a 7x9 block. Read literally as continuous rectangles
        those are zero area and 6x8. Nothing fails when this conversion is
        missed — the terrain just quietly gets smaller — which is why it lives at
        the single boundary rather than at each caller.
        """
        return cls(Polygon.from_cell_rect(x0, y0, x1, y1))

    @classmethod
    def from_outline(cls, points: list[tuple[float, float]]) -> "Footprint":
        """Build from explicit vertices, in board units and taken literally."""
        return cls(Polygon.from_points(points))


class Terrain:
    """Read-only collection of footprints, with the arrays the sight trace needs.

    The padded outline array is built once here rather than per query. A layout
    lives for a whole episode — `Battle.set_terrain` replaces it wholesale — so
    there is nothing to invalidate, and rebuilding it per sight pass would put an
    allocation in the hottest loop in the environment.
    """

    __slots__ = ("_footprints", "_outlines", "_vertex_counts")

    def __init__(self, footprints: list[Footprint]) -> None:
        self._footprints = footprints
        budget = max((f.n_vertices for f in footprints), default=0)
        if footprints:
            self._outlines = np.stack([f.polygon.padded_to(budget) for f in footprints])
            self._vertex_counts = np.array(
                [f.n_vertices for f in footprints], dtype=np.intp
            )
        else:
            self._outlines = np.zeros((0, 0, 2), dtype=float)
            self._vertex_counts = np.zeros(0, dtype=np.intp)

    @property
    def footprints(self) -> list[Footprint]:
        """All terrain footprints."""
        return self._footprints

    @property
    def outlines(self) -> np.ndarray:
        """``(n_pieces, vertex_budget, 2)`` vertices, padded by repetition."""
        return self._outlines

    @property
    def vertex_counts(self) -> np.ndarray:
        """``(n_pieces,)`` real vertex count per piece, excluding the padding."""
        return self._vertex_counts
