"""Polygon geometry for terrain footprints and terrain objectives.

Terrain and objectives used to be axis-aligned rectangles and circles. Both are now
polygons, so that a footprint can be any outline rather than a box.

Scope: this is the **outline** of a piece. The L- and U-shaped structures inside a ruin
are *walls*, a separate feature that does not exist yet -- see
``docs/rules/13-terrain.md``. Nothing here restricts a polygon to being convex, so
whichever way walls are eventually modelled, the outline will not be in the way.

Every test that runs per line-of-sight query is vectorised over sample points. Line of
sight is the hottest path in the environment and a Python loop over points would
dominate the step time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Polygon:
    """A simple (non-self-intersecting) polygon, closed implicitly.

    ``vertices`` is ``(n, 2)`` in board units, in order around the outline. The first
    vertex is not repeated at the end.
    """

    vertices: np.ndarray

    def __post_init__(self) -> None:
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 2:
            raise ValueError(
                f"polygon vertices must be (n, 2), got {self.vertices.shape}"
            )
        if len(self.vertices) < 3:
            raise ValueError(
                f"a polygon needs at least 3 vertices, got {len(self.vertices)}"
            )

    # -- constructors ----------------------------------------------------------

    @classmethod
    def from_points(cls, points: list[tuple[float, float]] | np.ndarray) -> "Polygon":
        """Build a polygon from a sequence of ``(x, y)`` points."""
        return cls(np.asarray(points, dtype=float).reshape(-1, 2))

    @classmethod
    def from_rect(cls, x0: float, y0: float, x1: float, y1: float) -> "Polygon":
        """Build an axis-aligned rectangle, normalising the corners."""
        lo_x, hi_x = min(x0, x1), max(x0, x1)
        lo_y, hi_y = min(y0, y1), max(y0, y1)
        return cls(
            np.array(
                [[lo_x, lo_y], [hi_x, lo_y], [hi_x, hi_y], [lo_x, hi_y]], dtype=float
            )
        )

    @classmethod
    def from_cell_rect(cls, x0: int, y0: int, x1: int, y1: int) -> "Polygon":
        """Build a rectangle from a corner-inclusive rectangle of whole cells.

        Configs author terrain as the *cells* a piece covers, so ``(5, 5, 5, 5)`` is one
        cell. Read literally as a continuous rectangle that would have zero area, so the
        far corner is pushed out by one and the authored area survives.
        """
        lo_x, hi_x = min(x0, x1), max(x0, x1)
        lo_y, hi_y = min(y0, y1), max(y0, y1)
        return cls.from_rect(float(lo_x), float(lo_y), float(hi_x + 1), float(hi_y + 1))

    # -- measurements ----------------------------------------------------------

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Axis-aligned bounding box as ``(x0, y0, x1, y1)``."""
        low = self.vertices.min(axis=0)
        high = self.vertices.max(axis=0)
        return (float(low[0]), float(low[1]), float(high[0]), float(high[1]))

    @property
    def area(self) -> float:
        """Enclosed area, by the shoelace formula. Sign-independent."""
        x, y = self.vertices[:, 0], self.vertices[:, 1]
        return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)

    @property
    def centroid(self) -> np.ndarray:
        """Area centroid. Falls back to the vertex mean for a degenerate polygon."""
        x, y = self.vertices[:, 0], self.vertices[:, 1]
        x_next, y_next = np.roll(x, -1), np.roll(y, -1)
        cross = x * y_next - x_next * y
        signed_area = float(cross.sum() / 2.0)
        if signed_area == 0.0:
            mean: np.ndarray = self.vertices.mean(axis=0)
            return mean
        cx = float(((x + x_next) * cross).sum() / (6.0 * signed_area))
        cy = float(((y + y_next) * cross).sum() / (6.0 * signed_area))
        return np.array([cx, cy], dtype=float)

    # -- point tests -----------------------------------------------------------

    def contains_points(self, points: np.ndarray) -> np.ndarray:
        """Vectorised point-in-polygon over an ``(..., 2)`` array of points.

        Crossing-number rule, which is correct for concave outlines as well as convex
        ones. Points exactly on an edge may fall either way; nothing here depends on
        that, and forcing a decision would cost a branch on the hot path.
        """
        flat = points.reshape(-1, 2)
        px = flat[:, 0][:, None]
        py = flat[:, 1][:, None]

        ax = self.vertices[:, 0][None, :]
        ay = self.vertices[:, 1][None, :]
        bx = np.roll(ax, -1, axis=1)
        by = np.roll(ay, -1, axis=1)

        # An edge is crossed when it straddles the point's y, and the crossing lies to
        # the right of the point.
        straddles = (ay > py) != (by > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at_py = (bx - ax) * (py - ay) / (by - ay) + ax
        crossings = straddles & (px < x_at_py)
        inside: np.ndarray = (crossings.sum(axis=1) % 2) == 1
        return inside.reshape(points.shape[:-1])

    def contains(self, x: float, y: float) -> bool:
        """True if ``(x, y)`` lies inside the polygon or on its boundary.

        Unlike the vectorised :meth:`contains_points`, this is inclusive of the edge.
        It decides endpoint membership -- whether a model is standing *in* a terrain
        piece, and so can see out of it -- and a model placed at exactly a corner by a
        fixed-position config would otherwise be judged outside and blocked by its own
        cover. The inclusive check is affordable here because it runs once per query,
        not once per sample along a ray.
        """
        point = np.array([x, y], dtype=float)
        if bool(self.contains_points(point)):
            return True
        return self._on_boundary(point)

    def _on_boundary(self, point: np.ndarray, tolerance: float = 1e-9) -> bool:
        """True if *point* lies on any edge, within *tolerance*."""
        starts = self.vertices
        ends = np.roll(self.vertices, -1, axis=0)
        edges = ends - starts
        lengths_squared = np.einsum("ed,ed->e", edges, edges)
        lengths_squared = np.where(lengths_squared == 0.0, 1.0, lengths_squared)
        t = np.clip(
            np.einsum("ed,ed->e", point - starts, edges) / lengths_squared, 0.0, 1.0
        )
        closest = starts + t[:, None] * edges
        return bool((np.linalg.norm(closest - point, axis=1) <= tolerance).any())

    def distance_to_point(self, point: np.ndarray) -> float:
        """Distance from *point* to the polygon, 0 when inside.

        Measured to the nearest edge, so it is the distance to the closest part of the
        shape -- which is how the rules measure to a terrain area.
        """
        if self.contains(float(point[0]), float(point[1])):
            return 0.0
        starts = self.vertices
        ends = np.roll(self.vertices, -1, axis=0)
        edges = ends - starts
        lengths_squared = np.einsum("ed,ed->e", edges, edges)
        lengths_squared = np.where(lengths_squared == 0.0, 1.0, lengths_squared)
        t = np.einsum("ed,ed->e", point - starts, edges) / lengths_squared
        t = np.clip(t, 0.0, 1.0)
        closest = starts + t[:, None] * edges
        return float(np.linalg.norm(closest - point, axis=1).min())

    # -- polygon-to-polygon ----------------------------------------------------

    def intersects(self, other: "Polygon") -> bool:
        """True if the two outlines share interior area, or one contains the other.

        Merely touching does not count. Two pieces authored as adjacent cell
        rectangles become continuous rectangles that share an edge, and treating that
        as an overlap would reject layouts that are plainly fine.

        Bounding boxes are checked first because most pairs are far apart and the box
        test rejects them for four comparisons. The test is strict, which is sound: if
        two shapes share any interior area then their boxes strictly overlap too.
        """
        ax0, ay0, ax1, ay1 = self.bounds
        bx0, by0, bx1, by1 = other.bounds
        if ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0:
            return False
        if _any_edges_cross(self.vertices, other.vertices):
            return True
        # No crossing edges, so either they are disjoint or one is wholly inside the
        # other; one vertex from each decides it.
        return bool(
            other.contains_points(self.vertices[0])
            or self.contains_points(other.vertices[0])
        )

    def mirrored(self, board_width: float) -> "Polygon":
        """Reflect the polygon across the board's vertical centre line.

        Winding is reversed by the reflection, so the vertex order is flipped back to
        keep every polygon consistently oriented.
        """
        flipped = self.vertices.copy()
        flipped[:, 0] = board_width - flipped[:, 0]
        return Polygon(flipped[::-1].copy())

    def padded_vertices(self, max_vertices: int) -> np.ndarray:
        """Vertices padded to ``max_vertices`` by repeating the last one.

        The observation stacks a whole batch of terrain, so the vertex count has to be
        fixed. Repeating the final vertex is geometrically inert -- it adds zero-length
        edges, which change neither the outline nor any test above.
        """
        if len(self.vertices) >= max_vertices:
            return self.vertices[:max_vertices].copy()
        pad = np.repeat(self.vertices[-1:], max_vertices - len(self.vertices), axis=0)
        return np.vstack([self.vertices, pad])


def _any_edges_cross(first: np.ndarray, second: np.ndarray) -> bool:
    """True if any edge of *first* intersects any edge of *second*."""
    a0 = first
    a1 = np.roll(first, -1, axis=0)
    b0 = second
    b1 = np.roll(second, -1, axis=0)

    # Broadcast every edge pair: (n_a, n_b, 2)
    p = a0[:, None, :]
    r = (a1 - a0)[:, None, :]
    q = b0[None, :, :]
    s = (b1 - b0)[None, :, :]

    denominator = _cross(r, s)
    qp = q - p
    with np.errstate(divide="ignore", invalid="ignore"):
        t = _cross(qp, s) / denominator
        u = _cross(qp, r) / denominator
    proper = (denominator != 0.0) & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)
    return bool(proper.any())


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2D cross product magnitude, broadcast over the leading axes."""
    result: np.ndarray = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return result
