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
        return bool(
            polygons_contain_points(
                point,
                self.vertices[np.newaxis, :, :],
                np.array([self.n_vertices]),
                include_boundary=True,
            )[0, 0]
        )

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

    def distance_to_point(self, x: float, y: float) -> float:
        """Distance from the point to the outline; 0.0 anywhere inside it."""
        if self.contains(x, y):
            return 0.0
        return self.distance_to_boundary(x, y)

    def distance_to_boundary(self, x: float, y: float) -> float:
        """Distance from the point to the nearest edge, inside or out.

        Unsigned, unlike `distance_to_point`, which reports 0.0 anywhere inside.
        A caller that has already established containment wants this one: it
        answers "how much room is left before a base of radius r crosses an
        edge", which is the rules' *wholly within* test.
        """
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

    def rotated_180(self, board_width: float, board_height: float) -> "Polygon":
        """Rotate a half turn about the board's centre.

        The point-symmetric partner to `mirrored`. Winding is *preserved*: a
        half turn is two reflections, so handedness survives and the vertex
        order does not need reversing — unlike a single reflection.
        """
        turned = self.vertices.copy()
        turned[:, 0] = board_width - turned[:, 0]
        turned[:, 1] = board_height - turned[:, 1]
        return Polygon(turned)

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


def closed_edge_ends(
    outlines: npt.NDArray[np.float64], vertex_counts: npt.NDArray[np.intp]
) -> npt.NDArray[np.float64]:
    """Edge endpoints for padded outlines, with every closing edge restored.

    Padding repeats the last real vertex, so a plain roll gives outline ``i`` a
    last real edge of ``v_last -> v_last`` — zero length, and therefore inert —
    while the true closing edge ``v_last -> v0`` lands in the slot the
    ``real_edge`` mask discards as padding. Both consumers below would then
    trace an *open* polyline: a crossing-number test gets the wrong parity in a
    wedge of the plane, and a distance test ignores one whole side.

    Only outlines shorter than the array's vertex budget are affected, which is
    why a fixture of uniform-width shapes cannot catch it.
    """
    ends: npt.NDArray[np.float64] = np.roll(outlines, -1, axis=1)
    if len(outlines):
        ends[np.arange(len(outlines)), vertex_counts - 1] = outlines[:, 0]
    return ends


def polygons_distance_to_points(
    points: npt.NDArray[np.float64],
    outlines: npt.NDArray[np.float64],
    vertex_counts: npt.NDArray[np.intp],
) -> npt.NDArray[np.float64]:
    """``(P, N)`` distance from each point to each outline; 0.0 anywhere inside.

    This is what makes an *area* objective work without a branch anywhere
    downstream. Give the area a radius of 0 and report this as the distance, and
    every existing ``norms_offset <= obj_radii`` test — in the reward, in VP
    scoring, in the success criteria — keeps its exact meaning: "close enough to
    count". No consumer needs to know objectives now have shapes.
    """
    n_points = len(points)
    n_outlines = len(outlines)
    if n_points == 0 or n_outlines == 0:
        return np.zeros((n_points, n_outlines), dtype=VERTEX_DTYPE)

    starts = outlines  # (N, V, 2)
    ends = closed_edge_ends(outlines, vertex_counts)
    edge = ends - starts  # (N, V, 2)
    length_sq = (edge**2).sum(axis=2)  # (N, V)
    safe_length = np.where(length_sq > 0, length_sq, 1.0)

    offset = points[:, np.newaxis, np.newaxis, :] - starts[np.newaxis, :, :, :]
    along = np.clip(
        (offset * edge[np.newaxis, :, :, :]).sum(axis=3) / safe_length, 0.0, 1.0
    )  # (P, N, V)
    closest = starts[np.newaxis, :, :, :] + along[..., np.newaxis] * edge[np.newaxis]
    distances = np.linalg.norm(
        closest - points[:, np.newaxis, np.newaxis, :], axis=3
    )  # (P, N, V)

    # Slots at or past the vertex count are padding and carry no edge. The real
    # edges are all below it *because* `closed_edge_ends` puts the closing edge
    # in the last real slot; masking alone would drop it.
    edge_index = np.arange(outlines.shape[1])
    real_edge = edge_index[np.newaxis, :] < vertex_counts[:, np.newaxis]
    distances = np.where(real_edge[np.newaxis, :, :], distances, np.inf)

    to_edge: npt.NDArray[np.float64] = distances.min(axis=2)
    inside = polygons_contain_points(
        points, outlines, vertex_counts, include_boundary=True
    )
    return np.where(inside, 0.0, to_edge)


def polygons_contain_points(
    points: npt.NDArray[np.float64],
    outlines: npt.NDArray[np.float64],
    vertex_counts: npt.NDArray[np.intp],
    *,
    include_boundary: bool = False,
    tolerance: float = 1e-9,
) -> npt.NDArray[np.bool_]:
    """``(P, N)`` membership of ``P`` points in ``N`` padded outlines, in one pass.

    Args:
        points: ``(P, 2)``.
        outlines: ``(N, V_max, 2)`` vertices, padded by repeating the last one.
        vertex_counts: ``(N,)`` real vertex count per outline, so the padded
            slots can be excluded. Pair it with `closed_edge_ends`, never with a
            bare roll: the mask discards the slot a roll puts the closing edge
            in, which traces the outline as an open polyline.
        include_boundary: also count points lying exactly on an edge. Off by
            default because sight *samples* are measure-zero and this doubles the
            work; on for anything where membership decides a rule, since a
            crossing-number test gives a boundary point whichever answer its
            edges happen to produce.

    Crossing number, not winding number, and vectorised **across shapes as well
    as points**. That is the whole performance story for sight: a per-piece
    python loop turns one query into dozens of tiny numpy calls whose overhead
    dwarfs the arithmetic, measured at 70.2 ms/step against 30.0 for a single
    padded pass.
    """
    n_points = len(points)
    n_outlines = len(outlines)
    inside = np.zeros((n_points, n_outlines), dtype=bool)
    if n_points == 0 or n_outlines == 0:
        return inside

    # Bounding-box pre-filter, and it is the difference between this being
    # affordable and not. The full test materialises (points x outlines x edges)
    # with a division in it; a box test is four comparisons and no allocation
    # per edge. On a 25v25 shooting phase against 37 pieces, well under 1% of
    # (sample, piece) pairs survive it -- LOS went from 7.9 ms/step to well
    # under 1 without changing a single answer.
    #
    # Padding repeats a real vertex, so min/max over the padded array is the
    # true bounding box and needs no masking.
    low = outlines.min(axis=1)  # (N, 2)
    high = outlines.max(axis=1)
    px_all = points[:, 0, np.newaxis]
    py_all = points[:, 1, np.newaxis]
    slack = tolerance if include_boundary else 0.0
    in_box = (
        (px_all >= low[:, 0] - slack)
        & (px_all <= high[:, 0] + slack)
        & (py_all >= low[:, 1] - slack)
        & (py_all <= high[:, 1] + slack)
    )  # (P, N)
    point_index, outline_index = np.nonzero(in_box)
    if len(point_index) == 0:
        return inside

    starts = outlines[outline_index]  # (K, V, 2)
    ends = closed_edge_ends(starts, vertex_counts[outline_index])
    edge_index = np.arange(outlines.shape[1])
    real_edge = edge_index[np.newaxis, :] < vertex_counts[outline_index][:, np.newaxis]

    px = points[point_index, 0][:, np.newaxis]  # (K, 1)
    py = points[point_index, 1][:, np.newaxis]
    x0 = starts[:, :, 0]  # (K, V)
    y0 = starts[:, :, 1]
    x1 = ends[:, :, 0]
    y1 = ends[:, :, 1]

    # A ray cast in +x crosses this edge iff the edge straddles the point's y
    # and the crossing lies to the right of it.
    straddles = (y0 > py) != (y1 > py)
    denominator = np.where(y1 != y0, y1 - y0, 1.0)
    crossing_x = x0 + (py - y0) * (x1 - x0) / denominator
    crosses = straddles & (px < crossing_x) & real_edge
    hit = (crosses.sum(axis=1) % 2) == 1

    if include_boundary:
        edge_x = x1 - x0
        edge_y = y1 - y0
        cross = edge_x * (py - y0) - edge_y * (px - x0)
        length_sq = edge_x**2 + edge_y**2
        safe_length = np.where(length_sq > 0, length_sq, 1.0)
        along = (edge_x * (px - x0) + edge_y * (py - y0)) / safe_length
        on_edge = (
            (np.abs(cross) <= tolerance * np.maximum(1.0, length_sq))
            & (along >= -tolerance)
            & (along <= 1.0 + tolerance)
            & real_edge
        )
        hit = hit | on_edge.any(axis=1)

    inside[point_index, outline_index] = hit
    return inside
