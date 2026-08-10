"""The pure-domain terrain model: outlines, their bounding boxes, and the arrays
the sight trace consumes.

A footprint is an outline now, not a bounding box. That distinction is the whole
reason the cover question was never fairly asked before: an L-shaped ruin and a
solid block reached both the sight trace and the network as the same four
numbers, so a policy could not have told them apart even in principle.
"""

from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs.domain.terrain import Footprint, Terrain
from wargame_rl.wargame.envs.types.geometry import Polygon


def test_a_footprint_is_edge_inclusive() -> None:
    """A model standing on a ruin's edge is standing *in* it, and can see out."""
    piece = Footprint.from_corners(0, 0, 2, 2)

    assert piece.contains(0, 0) is True
    assert piece.contains(2, 2) is True
    assert piece.contains(1, 1) is True
    assert piece.contains(3, 0) is False
    assert piece.contains(0, 3) is False


def test_from_corners_normalises_the_order() -> None:
    piece = Footprint.from_corners(2, 2, 0, 0)

    assert (piece.x0, piece.y0, piece.x1, piece.y1) == (0, 0, 2, 2)
    assert piece.contains(1, 1) is True
    assert piece.contains(3, 3) is False


def test_a_cell_rect_gains_its_far_corner() -> None:
    """`(5,5,5,5)` names one cell, so it covers a 1x1 area and not a point.

    Read literally as a continuous rectangle it has zero area, blocks nothing,
    and nothing fails — the terrain just quietly gets smaller.
    """
    piece = Footprint.from_cell_rect(5, 5, 5, 5)

    assert (piece.x0, piece.y0, piece.x1, piece.y1) == (5.0, 5.0, 6.0, 6.0)
    assert piece.contains(5.5, 5.5) is True


def test_the_bounding_box_is_a_summary_not_the_shape() -> None:
    """The defect polygon terrain exists to fix, stated as a test.

    (4, 4) is inside an L's bounding box and outside the L. Anything deciding
    whether a point is in a piece has to ask `contains`, not compare against
    x0..y1, or a concave ruin's notch counts as solid.
    """
    ell = Footprint.from_outline(
        [(0.0, 0.0), (6.0, 0.0), (6.0, 2.0), (2.0, 2.0), (2.0, 6.0), (0.0, 6.0)]
    )

    assert (ell.x0, ell.y0, ell.x1, ell.y1) == (0.0, 0.0, 6.0, 6.0)
    assert ell.contains(4.0, 4.0) is False
    assert ell.contains(1.0, 5.0) is True


def test_terrain_pads_outlines_to_a_common_vertex_budget() -> None:
    """Padding is what lets mixed shapes be tested in one array.

    Repeating the last vertex makes zero-length edges, which never register a
    crossing, so the padding cannot change an answer. `vertex_counts` keeps the
    real edge count available anyway.
    """
    triangle = Footprint(Polygon.from_points([(0.0, 0.0), (4.0, 0.0), (0.0, 4.0)]))
    square = Footprint.from_corners(10.0, 10.0, 14.0, 14.0)
    terrain = Terrain([triangle, square])

    assert terrain.outlines.shape == (2, 4, 2)
    assert terrain.vertex_counts.tolist() == [3, 4]
    # The triangle's padding repeats its final vertex.
    assert terrain.outlines[0][2].tolist() == terrain.outlines[0][3].tolist()


def test_empty_terrain_has_empty_arrays_rather_than_none() -> None:
    """A layout with no pieces still has to be indexable by the sight trace."""
    terrain = Terrain([])

    assert terrain.footprints == []
    assert terrain.outlines.shape[0] == 0
    assert terrain.vertex_counts.shape == (0,)


def test_distance_is_zero_inside_a_piece() -> None:
    piece = Footprint.from_corners(5.0, 5.0, 10.0, 10.0)

    assert piece.distance_to_point(7.0, 7.0) == 0.0
    assert piece.distance_to_point(13.0, 7.0) == 3.0


def test_touching_pieces_do_not_overlap() -> None:
    """Adjacent cell rectangles share an edge once the board is continuous."""
    left = Footprint.from_cell_rect(0, 0, 4, 4)
    right = Footprint.from_cell_rect(5, 0, 9, 4)

    assert left.overlaps(right) is False
    assert left.overlaps(Footprint.from_cell_rect(3, 0, 7, 4)) is True


def test_footprints_compare_by_outline() -> None:
    """Equality is the shape, so a mirrored pair can be checked for symmetry."""
    assert Footprint.from_corners(0, 0, 2, 2) == Footprint.from_corners(0, 0, 2, 2)
    assert Footprint.from_corners(0, 0, 2, 2) != Footprint.from_corners(0, 0, 3, 3)
    assert np.array_equal(
        Footprint.from_corners(0, 0, 2, 2).polygon.vertices,
        Polygon.from_rect(0.0, 0.0, 2.0, 2.0).vertices,
    )
