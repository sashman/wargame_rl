"""Polygon outlines for terrain footprints and terrain objectives."""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.envs.env_components.distance_cache import compute_distances
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.types.config import ObjectiveConfig, TerrainPieceConfig
from wargame_rl.wargame.envs.types.geometry import Polygon
from wargame_rl.wargame.envs.types.terrain_observation import MAX_TERRAIN_VERTICES
from wargame_rl.wargame.envs.wargame import WargameEnv

# An L, used only to exercise concave geometry. Concave *footprints* are not how walls
# inside a ruin will be modelled -- that is a separate feature -- but nothing in the
# geometry assumes convexity, and this pins that.
L_SHAPE = [(0.0, 0.0), (6.0, 0.0), (6.0, 2.0), (2.0, 2.0), (2.0, 6.0), (0.0, 6.0)]


class TestPolygon:
    def test_area_and_centroid_of_a_square(self) -> None:
        square = Polygon.from_rect(0.0, 0.0, 4.0, 4.0)

        assert square.area == pytest.approx(16.0)
        np.testing.assert_allclose(square.centroid, [2.0, 2.0])

    def test_area_ignores_winding_direction(self) -> None:
        """Shoelace is signed; area must not be."""
        clockwise = Polygon.from_points([(0, 0), (0, 3), (3, 3), (3, 0)])
        anticlockwise = Polygon.from_points([(0, 0), (3, 0), (3, 3), (0, 3)])

        assert clockwise.area == pytest.approx(anticlockwise.area)

    def test_concave_containment_excludes_the_notch(self) -> None:
        """The crossing-number rule has to handle a concave outline."""
        shape = Polygon.from_points(L_SHAPE)

        assert shape.contains(1.0, 1.0)  # in the corner of the L
        assert shape.contains(5.0, 1.0)  # along the foot
        assert not shape.contains(5.0, 5.0)  # in the notch

    def test_containment_includes_the_boundary(self) -> None:
        """A model standing on a piece's edge is in it, so it can see out of it."""
        square = Polygon.from_rect(0.0, 0.0, 2.0, 2.0)

        assert square.contains(2.0, 2.0)
        assert square.contains(0.0, 1.0)

    def test_distance_is_zero_inside_and_measured_to_the_edge_outside(self) -> None:
        square = Polygon.from_rect(0.0, 0.0, 4.0, 4.0)

        assert square.distance_to_point(np.array([2.0, 2.0])) == pytest.approx(0.0)
        assert square.distance_to_point(np.array([7.0, 2.0])) == pytest.approx(3.0)

    def test_touching_outlines_do_not_count_as_overlapping(self) -> None:
        """Adjacent cell rectangles share an edge once made continuous."""
        left = Polygon.from_rect(0.0, 0.0, 2.0, 2.0)
        right = Polygon.from_rect(2.0, 0.0, 4.0, 2.0)

        assert not left.intersects(right)

    def test_overlapping_outlines_are_detected(self) -> None:
        left = Polygon.from_rect(0.0, 0.0, 3.0, 3.0)
        right = Polygon.from_rect(2.0, 2.0, 5.0, 5.0)

        assert left.intersects(right)

    def test_containment_counts_as_intersection(self) -> None:
        """One outline wholly inside another crosses no edges."""
        outer = Polygon.from_rect(0.0, 0.0, 10.0, 10.0)
        inner = Polygon.from_rect(3.0, 3.0, 6.0, 6.0)

        assert outer.intersects(inner)
        assert inner.intersects(outer)

    def test_cell_rect_keeps_the_authored_area(self) -> None:
        """A one-cell piece is one unit square, not a zero-area rectangle."""
        assert Polygon.from_cell_rect(5, 5, 5, 5).area == pytest.approx(1.0)
        assert Polygon.from_cell_rect(27, 8, 33, 16).area == pytest.approx(7 * 9)

    def test_padding_is_geometrically_inert(self) -> None:
        """Repeated vertices must not change the shape the observation describes."""
        triangle = Polygon.from_points([(0, 0), (4, 0), (0, 4)])

        padded = Polygon(triangle.padded_vertices(MAX_TERRAIN_VERTICES))

        assert len(padded.vertices) == MAX_TERRAIN_VERTICES
        assert padded.area == pytest.approx(triangle.area)
        assert padded.contains(1.0, 1.0)
        assert not padded.contains(3.0, 3.0)

    def test_mirroring_is_its_own_inverse(self) -> None:
        shape = Polygon.from_points(L_SHAPE)

        there_and_back = shape.mirrored(60.0).mirrored(60.0)

        assert there_and_back.area == pytest.approx(shape.area)
        np.testing.assert_allclose(there_and_back.centroid, shape.centroid, atol=1e-9)

    def test_too_few_vertices_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 3 vertices"):
            Polygon.from_points([(0, 0), (1, 1)])


class TestPolygonTerrain:
    def test_a_polygon_piece_blocks_only_where_it_actually_is(self) -> None:
        """The outline blocks sight, not its bounding box.

        A concave piece is the sharpest test of this: a line through the notch passes
        through the bounding box but not through the piece.
        """
        env = WargameEnv(
            config=WargameEnvConfig(
                board_width=20,
                board_height=20,
                number_of_wargame_models=1,
                number_of_objectives=1,
                number_of_battle_rounds=1,
                render_mode=None,
                # An L occupying x 5..11, y 5..11 with its notch at the top right.
                terrain=[
                    TerrainPieceConfig(
                        polygon=[
                            (5, 5),
                            (11, 5),
                            (11, 7),
                            (7, 7),
                            (7, 11),
                            (5, 11),
                        ]
                    )
                ],
            )
        )

        # Through the foot of the L: blocked.
        assert env.has_line_of_sight_between_cells(0, 6, 19, 6) is False
        # Through the notch, which the bounding box covers but the outline does not.
        assert env.has_line_of_sight_between_cells(8, 19, 8, 9) is True

    def test_a_config_may_not_give_both_a_rectangle_and_an_outline(self) -> None:
        with pytest.raises(ValueError, match="exactly one of footprint, polygon"):
            TerrainPieceConfig(footprint=(0, 0, 2, 2), polygon=L_SHAPE)

    def test_a_config_must_give_one_or_the_other(self) -> None:
        with pytest.raises(ValueError, match="exactly one of footprint, polygon"):
            TerrainPieceConfig()

    def test_overlapping_polygon_pieces_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="overlaps"):
            WargameEnvConfig(
                board_width=30,
                board_height=30,
                number_of_wargame_models=1,
                number_of_objectives=1,
                render_mode=None,
                terrain=[
                    TerrainPieceConfig(polygon=[(5, 5), (15, 5), (15, 15), (5, 15)]),
                    TerrainPieceConfig(
                        polygon=[(10, 10), (20, 10), (20, 20), (10, 20)]
                    ),
                ],
            )

    def test_adjacent_polygon_pieces_are_allowed(self) -> None:
        """Sharing an edge is not overlapping."""
        config = WargameEnvConfig(
            board_width=30,
            board_height=30,
            number_of_wargame_models=1,
            number_of_objectives=1,
            render_mode=None,
            terrain=[
                TerrainPieceConfig(polygon=[(5, 5), (10, 5), (10, 15), (5, 15)]),
                TerrainPieceConfig(polygon=[(10, 5), (15, 5), (15, 15), (10, 15)]),
            ],
        )

        assert config.terrain is not None


class TestPolygonObjective:
    def _env(self) -> WargameEnv:
        return WargameEnv(
            config=WargameEnvConfig(
                board_width=30,
                board_height=30,
                number_of_wargame_models=1,
                number_of_objectives=1,
                number_of_battle_rounds=1,
                render_mode=None,
                objectives=[
                    ObjectiveConfig(polygon=[(10, 10), (20, 10), (20, 20), (10, 20)])
                ],
            )
        )

    def test_the_area_is_the_objective(self) -> None:
        """A model is in range while its base overlaps the area, not a disc."""
        env = self._env()
        env.reset(seed=0)
        model = env.wargame_models[0]

        model.location = np.array([15.0, 15.0])  # well inside
        assert self._in_range(env)

        model.location = np.array([25.0, 15.0])  # five units clear of the edge
        assert not self._in_range(env)

    def test_the_base_edge_reaches_into_the_area(self) -> None:
        """Just outside by less than a base radius still counts."""
        env = self._env()
        env.reset(seed=0)
        model = env.wargame_models[0]

        model.location = np.array([20.0 + model.base_radius / 2.0, 15.0])

        assert self._in_range(env)

    def test_the_objective_reports_the_area_centroid_as_its_position(self) -> None:
        """Anything steering toward an objective still has a point to aim at."""
        env = self._env()
        env.reset(seed=0)

        assert env.objectives[0].is_area
        np.testing.assert_allclose(env.objectives[0].location, [15.0, 15.0])

    def test_a_polygon_objective_counts_as_a_fixed_position(self) -> None:
        """The outline pins it, so random placement must not move it."""
        env = self._env()
        env.reset(seed=3)

        np.testing.assert_allclose(env.objectives[0].location, [15.0, 15.0])

    @staticmethod
    def _in_range(env: WargameEnv) -> bool:
        cache = compute_distances(env.wargame_models, env.objectives)
        return bool(cache.model_obj_norms_offset[0, 0] <= cache.obj_radii[0])
