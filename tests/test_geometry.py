"""Polygon geometry: the shape primitive under terrain and area objectives.

The two asymmetric boundary decisions are the interesting part and are pinned
here: `contains` is edge-inclusive so a model on a ruin's corner is standing *in*
it and can see out, while `contains_points` is interior-only because it sits on
the sight hot path; and touching is not overlapping, because adjacent cell
rectangles share an edge once the board is continuous.
"""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.envs.types.geometry import Polygon, polygons_contain_points

# An L, wound anticlockwise. Concave, so a convex half-plane test gets it wrong.
L_SHAPE = Polygon.from_points(
    [(0.0, 0.0), (6.0, 0.0), (6.0, 2.0), (2.0, 2.0), (2.0, 6.0), (0.0, 6.0)]
)


class TestConstruction:
    def test_a_polygon_needs_three_vertices(self) -> None:
        with pytest.raises(ValueError, match="at least 3 vertices"):
            Polygon.from_points([(0.0, 0.0), (1.0, 1.0)])

    def test_the_wrong_shape_is_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match=r"\(n_vertices, 2\)"):
            Polygon(np.zeros((4, 3)))

    def test_a_cell_rect_gains_its_far_corner(self) -> None:
        """`(5,5,5,5)` names one cell, which is a 1x1 area and not a point.

        Read literally as a continuous rectangle it has zero area, and nothing
        would fail — the terrain would just quietly get smaller.
        """
        assert Polygon.from_cell_rect(5, 5, 5, 5).area == pytest.approx(1.0)
        assert Polygon.from_cell_rect(27, 8, 33, 16).area == pytest.approx(7.0 * 9.0)


class TestMeasurement:
    def test_area_ignores_winding_order(self) -> None:
        clockwise = Polygon.from_points(
            [(0.0, 0.0), (0.0, 4.0), (3.0, 4.0), (3.0, 0.0)]
        )
        assert clockwise.area == pytest.approx(12.0)
        assert L_SHAPE.area == pytest.approx(6 * 2 + 2 * 4)

    def test_the_centroid_is_area_weighted_not_the_vertex_mean(self) -> None:
        """The distinction matters: this is what an area objective reports as its
        location, and every scripted policy steers at it."""
        assert L_SHAPE.centroid != pytest.approx(L_SHAPE.vertices.mean(axis=0))
        square = Polygon.from_rect(0.0, 0.0, 4.0, 4.0)
        assert square.centroid == pytest.approx([2.0, 2.0])

    def test_distance_is_zero_inside_and_perpendicular_outside(self) -> None:
        square = Polygon.from_rect(0.0, 0.0, 4.0, 4.0)
        assert square.distance_to_point(2.0, 2.0) == 0.0
        assert square.distance_to_point(7.0, 2.0) == pytest.approx(3.0)
        assert square.distance_to_point(-3.0, -4.0) == pytest.approx(5.0)


class TestMembership:
    def test_a_concave_outline_excludes_its_own_notch(self) -> None:
        """The reason this is crossing-number and not a convex test.

        (4, 4) is inside the L's bounding box and inside its convex hull, and
        outside the L itself.
        """
        assert L_SHAPE.contains(1.0, 1.0) is True
        assert L_SHAPE.contains(5.0, 1.0) is True
        assert L_SHAPE.contains(4.0, 4.0) is False

    def test_contains_is_edge_inclusive_where_contains_points_is_arbitrary(
        self,
    ) -> None:
        """Deliberately asymmetric, and the asymmetry is the reason for two methods.

        A model placed exactly on a corner by a fixed config must count as
        standing *in* the piece, or it is blocked by its own cover — so anything
        deciding a *rule* has to use `contains`. A crossing-number test gives a
        boundary point whichever answer its edges happen to produce: on this
        square the bottom edge reads inside and the top edge outside. That is
        fine for sight samples, which are measure-zero and on the hot path, and
        it is exactly why they are separate calls.
        """
        square = Polygon.from_rect(0.0, 0.0, 4.0, 4.0)
        boundary = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 4.0], [4.0, 2.0]])

        assert all(square.contains(float(x), float(y)) for x, y in boundary)
        assert not square.contains_points(boundary).all()

    def test_many_points_against_many_padded_outlines_in_one_pass(self) -> None:
        """Padding by repeating a vertex must not change any answer.

        Zero-length edges are what make outlines of different sizes shareable in
        one array, and they are only free if they never register a crossing.
        """
        triangle = Polygon.from_points([(0.0, 0.0), (4.0, 0.0), (0.0, 4.0)])
        square = Polygon.from_rect(10.0, 10.0, 14.0, 14.0)
        budget = 6
        outlines = np.stack([triangle.padded_to(budget), square.padded_to(budget)])
        counts = np.array([triangle.n_vertices, square.n_vertices])

        points = np.array([[1.0, 1.0], [12.0, 12.0], [50.0, 50.0], [3.0, 3.0]])
        inside = polygons_contain_points(points, outlines, counts)

        assert inside.tolist() == [
            [True, False],
            [False, True],
            [False, False],
            [False, False],  # (3,3) is outside the triangle's hypotenuse
        ]


class TestOverlapAndMirroring:
    def test_touching_is_not_overlapping(self) -> None:
        """Adjacent cell rectangles share an edge once the board is continuous.

        Treating that as an overlap rejects layouts that are plainly fine — and
        `from_cell_rect` produces exactly this case for any two neighbouring
        pieces.
        """
        left = Polygon.from_cell_rect(0, 0, 4, 4)
        right = Polygon.from_cell_rect(5, 0, 9, 4)
        assert left.overlaps(right) is False

    def test_a_genuine_overlap_is_caught(self) -> None:
        assert (
            Polygon.from_rect(0.0, 0.0, 5.0, 5.0).overlaps(
                Polygon.from_rect(4.0, 4.0, 9.0, 9.0)
            )
            is True
        )

    def test_mirroring_is_about_the_width_not_the_last_index(self) -> None:
        """`width - 1` was the last cell index and shifts every piece one unit left.

        Nothing fails when this is wrong: the layout just stops being symmetric,
        which reads as a generator bug rather than an off-by-one.
        """
        piece = Polygon.from_cell_rect(2, 3, 6, 7)
        mirrored = piece.mirrored(40.0)

        x0, y0, x1, y1 = mirrored.bounds
        assert (x0, x1) == pytest.approx((40.0 - 7.0, 40.0 - 2.0))
        assert (y0, y1) == pytest.approx((3.0, 8.0))

    def test_mirroring_twice_is_the_identity(self) -> None:
        piece = Polygon.from_points([(1.0, 2.0), (7.0, 3.0), (5.0, 9.0)])
        assert piece.mirrored(40.0).mirrored(40.0).bounds == pytest.approx(piece.bounds)


class TestPadding:
    def test_padding_repeats_the_last_vertex(self) -> None:
        triangle = Polygon.from_points([(0.0, 0.0), (4.0, 0.0), (0.0, 4.0)])
        padded = triangle.padded_to(6)

        assert padded.shape == (6, 2)
        assert padded[3:].tolist() == [[0.0, 4.0], [0.0, 4.0], [0.0, 4.0]]

    def test_padding_below_the_real_count_is_an_error(self) -> None:
        """Silently dropping vertices would turn a ruin into a smaller ruin."""
        with pytest.raises(ValueError, match="Cannot pad"):
            L_SHAPE.padded_to(4)
