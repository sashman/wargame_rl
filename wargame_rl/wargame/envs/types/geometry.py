"""Polygon geometry: the shape primitive shared by config, domain and rendering.

**This lives in `types/`, not `domain/`, and that is forced rather than
preferred.** `types/config/terrain.py` validates outlines at load time and cannot
import from `domain/` without inverting the dependency direction the whole env
layering rests on (see `docs/ddd-envs.md`). `types/` is the shared kernel; a
shape is exactly the kind of thing that belongs in it.

Two boundary decisions worth stating once, because they are asymmetric on
purpose:

- **`contains` is edge-inclusive; `contains_points` is not.** Endpoint membership
  decides whether a model is standing *in* a piece and can therefore see out of
  it, so a model placed exactly on a corner by a fixed config must not be
  blocked by its own cover. Sample membership is measure-zero and sits on the
  hot path, so it stays cheap.
- **Touching is not overlapping.** Once the board is continuous, two adjacent
  cell rectangles share an edge, and treating that as an overlap rejects layouts
  that are plainly fine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

# Shared coordinate type with `domain/value_objects.py`. Declared separately
# rather than imported, because `types/` must not depend on `domain/` -- the
# whole reason this module is here.
VERTEX_DTYPE = np.float64


@dataclass(frozen=True, slots=True)
class Polygon:
    """A closed outline as ``(n_vertices, 2)`` vertices, in board units.

    Vertices are stored in the order given. Nothing assumes convexity: the
    crossing-number test works on concave outlines, which is what lets an
    L-shaped ruin be a single piece.
    """

    vertices: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        array = np.asarray(self.vertices, dtype=VERTEX_DTYPE)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError(f"A polygon is (n_vertices, 2); got shape {array.shape}")
        if len(array) < 3:
            raise ValueError(f"A polygon needs at least 3 vertices, got {len(array)}")
        object.__setattr__(self, "vertices", array)

    @classmethod
    def from_points(cls, points: list[tuple[float, float]]) -> "Polygon":
        """Build from a list of ``(x, y)`` pairs."""
        return cls(np.array(points, dtype=VERTEX_DTYPE))

    @classmethod
    def from_rect(cls, x0: float, y0: float, x1: float, y1: float) -> "Polygon":
        """Build the axis-aligned rectangle with those corners, wound anticlockwise."""
        left, right = min(x0, x1), max(x0, x1)
        bottom, top = min(y0, y1), max(y0, y1)
        return cls.from_points(
            [(left, bottom), (right, bottom), (right, top), (left, top)]
        )

    @classmethod
    def from_cell_rect(cls, x0: int, y0: int, x1: int, y1: int) -> "Polygon":
        """Build from an inclusive *cell* rectangle, pushing the far corner out by one.

        The same corner-inclusive convention as `Footprint.from_cell_rect`:
        ``(5, 5, 5, 5)`` names one cell, which is a 1x1 area and not a point.
        """
        return cls.from_rect(
            float(min(x0, x1)),
            float(min(y0, y1)),
            float(max(x0, x1)) + 1.0,
            float(max(y0, y1)) + 1.0,
        )

    @property
    def n_vertices(self) -> int:
        """Number of vertices in the outline."""
        return int(len(self.vertices))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Axis-aligned bounding box as ``(x0, y0, x1, y1)``."""
        low = self.vertices.min(axis=0)
        high = self.vertices.max(axis=0)
        return (float(low[0]), float(low[1]), float(high[0]), float(high[1]))

    @property
    def area(self) -> float:
        """Absolute area, by the shoelace formula. Winding-order independent."""
        x = self.vertices[:, 0]
        y = self.vertices[:, 1]
        return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)

    @property
    def centroid(self) -> npt.NDArray[np.float64]:
        """Area centroid — the point an area objective reports as its location.

        Falls back to the vertex mean for a degenerate (zero-area) outline,
        where the area-weighted formula divides by zero.
        """
        x = self.vertices[:, 0]
        y = self.vertices[:, 1]
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)
        cross = x * y_next - x_next * y
        signed_area = float(cross.sum() / 2.0)
        if abs(signed_area) < 1e-12:
            mean: npt.NDArray[np.float64] = self.vertices.mean(axis=0)
            return mean
        return np.array(
            [
                float(np.dot(x + x_next, cross) / (6.0 * signed_area)),
                float(np.dot(y + y_next, cross) / (6.0 * signed_area)),
            ],
            dtype=VERTEX_DTYPE,
        )

    def contains(self, x: float, y: float) -> bool:
        """True if the point is inside or exactly on the outline. Edge-inclusive."""
        point = np.array([[x, y]], dtype=VERTEX_DTYPE)
        if self._points_on_boundary(point)[0]:
            return True
        return bool(self.contains_points(point)[0])

    def contains_points(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        """``(P,)`` membership for ``(P, 2)`` points, by crossing number.

        Interior only — boundary membership is deliberately unspecified here and
        is the caller's business via `contains`. Works on concave outlines, which
        a convex half-plane test would not.
        """
        points = np.asarray(points, dtype=VERTEX_DTYPE)
        if len(points) == 0:
            return np.zeros(0, dtype=bool)
        return polygons_contain_points(
            points, self.vertices[np.newaxis, :, :], np.array([self.n_vertices])
        )[:, 0]

    def _points_on_boundary(
        self, points: npt.NDArray[np.float64], tolerance: float = 1e-12
    ) -> npt.NDArray[np.bool_]:
        """``(P,)`` — True where a point lies on an edge, within *tolerance*."""
        starts = self.vertices
        ends = np.roll(self.vertices, -1, axis=0)
        edge = ends - starts  # (V, 2)
        offset = points[:, np.newaxis, :] - starts[np.newaxis, :, :]  # (P, V, 2)
        cross = edge[:, 0] * offset[:, :, 1] - edge[:, 1] * offset[:, :, 0]
        edge_length_sq = (edge**2).sum(axis=1)
        # Projection parameter along the edge, guarded for a zero-length edge --
        # padding to a vertex budget creates those on purpose.
        safe_length = np.where(edge_length_sq > 0, edge_length_sq, 1.0)
        along = (offset * edge[np.newaxis, :, :]).sum(axis=2) / safe_length
        on_segment = (along >= -tolerance) & (along <= 1.0 + tolerance)
        collinear = np.abs(cross) <= tolerance * np.maximum(1.0, edge_length_sq)
        result: npt.NDArray[np.bool_] = (collinear & on_segment).any(axis=1)
        return result

    def distance_to_point(self, x: float, y: float) -> float:
        """Distance from the point to the outline; 0.0 anywhere inside it."""
        if self.contains(x, y):
            return 0.0
        point = np.array([x, y], dtype=VERTEX_DTYPE)
        starts = self.vertices
        ends = np.roll(self.vertices, -1, axis=0)
        edge = ends - starts
        edge_length_sq = (edge**2).sum(axis=1)
        safe_length = np.where(edge_length_sq > 0, edge_length_sq, 1.0)
        along = np.clip(((point - starts) * edge).sum(axis=1) / safe_length, 0.0, 1.0)
        closest = starts + along[:, np.newaxis] * edge
        return float(np.linalg.norm(closest - point, axis=1).min())

    def overlaps(self, other: "Polygon") -> bool:
        """True if the two outlines share interior area. Touching does not count.

        Uses the separating-axis theorem, which is exact for convex outlines and
        conservative (may report an overlap that is not one) for concave ones.
        Terrain generation produces convex n-gons, and a conservative answer at
        config load is the safe direction: it rejects a layout rather than
        shipping one with two pieces fused into an accidental wall.
        """
        return not (
            _has_separating_axis(self.vertices, other.vertices)
            or _has_separating_axis(other.vertices, self.vertices)
        )

    def mirrored(self, board_width: float) -> "Polygon":
        """Reflect across the board's vertical centre line.

        About `board_width`, never `board_width - 1`. The `-1` was the last cell
        *index*; reflecting a continuous outline about it shifts every piece one
        unit left, which reads as an asymmetric layout rather than as an
        off-by-one.

        Winding order is reversed so the mirror image keeps the same handedness.
        """
        flipped = self.vertices.copy()
        flipped[:, 0] = board_width - flipped[:, 0]
        return Polygon(flipped[::-1].copy())

    def padded_to(self, n_vertices: int) -> npt.NDArray[np.float64]:
        """``(n_vertices, 2)`` with the last vertex repeated to fill.

        Padding by repetition is free for both consumers: a repeated vertex makes
        a zero-length edge, which never straddles a sample in the crossing-number
        test and contributes nothing to the observation beyond a duplicate. That
        is what lets outlines of different sizes be tested — and batched — in one
        array.
        """
        if n_vertices < self.n_vertices:
            raise ValueError(
                f"Cannot pad a {self.n_vertices}-vertex polygon down to {n_vertices}"
            )
        if n_vertices == self.n_vertices:
            return self.vertices.copy()
        tail = np.repeat(self.vertices[-1:], n_vertices - self.n_vertices, axis=0)
        return np.vstack([self.vertices, tail])


def _has_separating_axis(
    subject: npt.NDArray[np.float64], other: npt.NDArray[np.float64]
) -> bool:
    """True if some edge normal of *subject* separates the two vertex sets.

    Shared edges project to touching-but-not-overlapping intervals, so the
    comparison is strict: `>=` rather than `>`.
    """
    edges = np.roll(subject, -1, axis=0) - subject
    normals = np.column_stack([-edges[:, 1], edges[:, 0]])
    lengths = np.linalg.norm(normals, axis=1)
    normals = normals[lengths > 0]
    if len(normals) == 0:
        return False
    subject_projections = subject @ normals.T  # (V_subject, A)
    other_projections = other @ normals.T  # (V_other, A)
    separated = (subject_projections.min(axis=0) >= other_projections.max(axis=0)) | (
        other_projections.min(axis=0) >= subject_projections.max(axis=0)
    )
    return bool(separated.any())


def polygons_contain_points(
    points: npt.NDArray[np.float64],
    outlines: npt.NDArray[np.float64],
    vertex_counts: npt.NDArray[np.intp],
) -> npt.NDArray[np.bool_]:
    """``(P, N)`` membership of ``P`` points in ``N`` padded outlines, in one pass.

    Args:
        points: ``(P, 2)``.
        outlines: ``(N, V_max, 2)`` vertices, padded by repeating the last one.
        vertex_counts: ``(N,)`` real vertex count per outline, so the padded
            edges can be excluded. A padded edge is zero-length and would never
            be crossed anyway; excluding it explicitly costs one comparison and
            removes the need to reason about that.

    Crossing number, not winding number, and vectorised **across shapes as well
    as points**. That is the whole performance story for sight: a per-piece
    python loop turns one query into dozens of tiny numpy calls whose overhead
    dwarfs the arithmetic, measured at 70.2 ms/step against 30.0 for a single
    padded pass.
    """
    n_points = len(points)
    n_outlines = len(outlines)
    if n_points == 0 or n_outlines == 0:
        return np.zeros((n_points, n_outlines), dtype=bool)

    starts = outlines  # (N, V, 2)
    ends = np.roll(outlines, -1, axis=1)  # (N, V, 2)

    edge_index = np.arange(outlines.shape[1])
    real_edge = edge_index[np.newaxis, :] < vertex_counts[:, np.newaxis]  # (N, V)

    px = points[:, np.newaxis, np.newaxis, 0]  # (P, 1, 1)
    py = points[:, np.newaxis, np.newaxis, 1]
    x0 = starts[np.newaxis, :, :, 0]  # (1, N, V)
    y0 = starts[np.newaxis, :, :, 1]
    x1 = ends[np.newaxis, :, :, 0]
    y1 = ends[np.newaxis, :, :, 1]

    # A ray cast in +x crosses this edge iff the edge straddles the point's y
    # and the crossing lies to the right of it.
    straddles = (y0 > py) != (y1 > py)
    denominator = np.where(y1 != y0, y1 - y0, 1.0)
    crossing_x = x0 + (py - y0) * (x1 - x0) / denominator
    crosses = straddles & (px < crossing_x) & real_edge[np.newaxis, :, :]
    inside: npt.NDArray[np.bool_] = (crosses.sum(axis=2) % 2) == 1
    return inside
