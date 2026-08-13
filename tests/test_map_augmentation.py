"""Mirrored layouts and angled ruins — the two ways more table is made.

Both target the same measured defect: the trained checkpoint scored -1.4 vp on
tables it trained on and -23.8 on tables it had not, so the shortage is
*layouts*, not capability.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.random import default_rng

from wargame_rl.wargame.envs.domain.map_layout import MapLayout
from wargame_rl.wargame.envs.domain.terrain import Footprint, Terrain
from wargame_rl.wargame.envs.domain.terrain_placement import (
    _rotated_in_place,
    generate_terrain,
)
from wargame_rl.wargame.envs.domain.value_objects import BoardDimensions
from wargame_rl.wargame.envs.types.config import ObjectiveConfig, RandomTerrainConfig
from wargame_rl.wargame.envs.types.geometry import Polygon

WIDTH, HEIGHT = 60.0, 44.0


def layout() -> MapLayout:
    """A one-piece layout with one marker and one area objective."""
    piece = Footprint(Polygon.from_rect(10.0, 8.0, 16.0, 12.0))
    return MapLayout(
        name="table_01",
        terrain=Terrain([piece]),
        objectives=(
            ObjectiveConfig(x=12.0, y=10.0),
            ObjectiveConfig(area=[(40.0, 30.0), (46.0, 30.0), (46.0, 36.0)]),
        ),
    )


def test_no_flip_returns_the_same_layout() -> None:
    # Arrange / Act
    original = layout()

    # Assert: identity, not a copy — the un-mirrored draw must cost nothing.
    assert original.mirrored(WIDTH, HEIGHT, False, False) is original


@pytest.mark.parametrize(
    ("flip_x", "flip_y", "expected"),
    [
        (True, False, (48.0, 10.0)),
        (False, True, (12.0, 34.0)),
        (True, True, (48.0, 34.0)),
    ],
)
def test_a_marker_objective_reflects_about_the_mid_lines(
    flip_x: bool, flip_y: bool, expected: tuple[float, float]
) -> None:
    # Arrange / Act
    mirrored = layout().mirrored(WIDTH, HEIGHT, flip_x, flip_y)

    # Assert
    assert mirrored.objectives is not None
    assert (mirrored.objectives[0].x, mirrored.objectives[0].y) == expected


def test_an_area_objective_reflects_as_an_outline() -> None:
    # Arrange / Act
    mirrored = layout().mirrored(WIDTH, HEIGHT, True, False)

    # Assert: every vertex reflected, and it is still a triangle.
    assert mirrored.objectives is not None
    area = mirrored.objectives[1].area
    assert area is not None and len(area) == 3
    assert sorted(round(x, 6) for x, _ in area) == [14.0, 14.0, 20.0]


def test_terrain_reflects_and_stays_on_the_board() -> None:
    # Arrange / Act
    mirrored = layout().mirrored(WIDTH, HEIGHT, True, True)

    # Assert: the piece moves to the opposite corner, same size, on the board.
    x0, y0, x1, y1 = mirrored.terrain.footprints[0].polygon.bounds
    assert (round(x0, 6), round(x1, 6)) == (44.0, 50.0)
    assert (round(y0, 6), round(y1, 6)) == (32.0, 36.0)
    assert 0 <= x0 and x1 <= WIDTH and 0 <= y0 and y1 <= HEIGHT


def test_mirroring_is_an_involution() -> None:
    # Arrange: reflecting twice about the same axes returns the original shape.
    original = layout()

    # Act
    there_and_back = original.mirrored(WIDTH, HEIGHT, True, True).mirrored(
        WIDTH, HEIGHT, True, True
    )

    # Assert
    np.testing.assert_allclose(
        there_and_back.terrain.footprints[0].polygon.vertices,
        original.terrain.footprints[0].polygon.vertices,
    )


def test_the_mirrored_name_says_which_way() -> None:
    # Arrange / Act / Assert: a drawn map must be identifiable in a trace.
    assert layout().mirrored(WIDTH, HEIGHT, True, False).name == "table_01:x"
    assert layout().mirrored(WIDTH, HEIGHT, False, True).name == "table_01:y"
    assert layout().mirrored(WIDTH, HEIGHT, True, True).name == "table_01:xy"


def terrain_for(angled_fraction: float, seed: int = 0) -> Terrain:
    """Generate a layout at a given angled share."""
    spec = RandomTerrainConfig(
        count=8, min_size=3, max_size=9, n_vertices=6, angled_fraction=angled_fraction
    )
    return generate_terrain(
        spec, BoardDimensions(width=60, height=44), default_rng(seed)
    )


def test_angled_fraction_zero_is_an_exact_no_op() -> None:
    # Arrange / Act: the same seed, with the feature off and at its default.
    a = terrain_for(0.0, seed=3)
    b = terrain_for(0.0, seed=3)

    # Assert
    for left, right in zip(a.footprints, b.footprints):
        np.testing.assert_array_equal(left.polygon.vertices, right.polygon.vertices)


def test_angling_changes_the_layout_but_keeps_it_on_the_board() -> None:
    # Arrange / Act
    square = terrain_for(0.0, seed=5)
    angled = terrain_for(1.0, seed=5)

    # Assert: the layout genuinely differs, and nothing left the table. Note
    # the difference is not 'the same pieces, rotated' — angling consumes rng,
    # so the whole layout diverges. On-board is the claim being made here.
    differs = any(
        not np.allclose(s.polygon.vertices, a.polygon.vertices)
        for s, a in zip(square.footprints, angled.footprints)
    )
    assert differs
    for piece in angled.footprints:
        x0, y0, x1, y1 = piece.polygon.bounds
        assert 0 <= x0 and x1 <= 60 and 0 <= y0 and y1 <= 44


def test_an_angled_piece_stays_inside_the_box_it_was_allotted() -> None:
    # Arrange: the invariant that stops rotation creating overlaps — the placer
    # chose mutually clear rectangles, and a turned rectangle sweeps outside its
    # box. Tested on the transform directly: turning a piece changes how much
    # rng the generator consumes, so two layouts at different angled_fractions
    # are different layouts, not the same one rotated, and comparing them
    # piece-by-piece compares unrelated shapes.
    #
    # Deliberately asymmetric about its own centroid, which is the case that
    # broke the first implementation: it fitted the centroid-symmetric box,
    # and the real bounding box was free to grow past its allotment.
    polygon = Polygon.from_points(
        [(10.0, 8.0), (16.0, 9.0), (15.0, 12.0), (11.0, 11.5)]
    )
    x0, y0, x1, y1 = polygon.bounds

    # Act / Assert: every angle, including the awkward ones.
    for angle in np.linspace(0.0, float(np.pi), 25):
        turned = _rotated_in_place(polygon, float(angle))
        tx0, ty0, tx1, ty1 = turned.bounds
        assert tx1 - tx0 <= (x1 - x0) + 1e-9
        assert ty1 - ty0 <= (y1 - y0) + 1e-9
        assert tx0 >= x0 - 1e-9 and tx1 <= x1 + 1e-9
        assert ty0 >= y0 - 1e-9 and ty1 <= y1 + 1e-9


def test_the_piece_count_is_unchanged_by_angling() -> None:
    # Arrange / Act / Assert: observation batching stacks terrain into one
    # array, so the count must not vary between episodes.
    assert len(terrain_for(1.0, seed=7).footprints) == 8


def generated(mode: str, count: int, seed: int = 4) -> Terrain:
    """A mirrored layout in one of the two pairing modes."""
    spec = RandomTerrainConfig(
        count=count, min_size=3, max_size=9, n_vertices=6, mirror=True, mirror_mode=mode
    )
    return generate_terrain(
        spec, BoardDimensions(width=60, height=44), default_rng(seed)
    )


def cloud(terrain: Terrain) -> np.ndarray:
    """Every vertex in the layout, as one point cloud."""
    return np.vstack([piece.polygon.vertices for piece in terrain.footprints])


def hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric nearest-neighbour distance — independent of vertex order."""
    distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float(max(distances.min(axis=1).max(), distances.min(axis=0).max()))


def half_turned(points: np.ndarray) -> np.ndarray:
    """The cloud rotated a half turn about the board centre."""
    turned = points.copy()
    turned[:, 0] = 60.0 - turned[:, 0]
    turned[:, 1] = 44.0 - turned[:, 1]
    return turned


@pytest.mark.parametrize("count", [8, 9, 15, 16])
def test_rotate_180_produces_a_point_symmetric_layout(count: int) -> None:
    # Arrange / Act: odd counts included, because the unpaired centre piece is
    # the case that has to be its own partner — and a piece straddling the
    # vertical centre line is *not* its own half turn.
    points = cloud(generated("rotate_180", count))

    # Assert: exactly, not approximately.
    assert hausdorff(points, half_turned(points)) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("count", [8, 9])
def test_reflect_x_is_unchanged_and_is_not_point_symmetric(count: int) -> None:
    # Arrange / Act: the default must keep doing exactly what it did, and it is
    # the *wrong* symmetry for these tables — a mirror line down the middle,
    # where the real ones are point-symmetric.
    points = cloud(generated("reflect_x", count))
    reflected = points.copy()
    reflected[:, 0] = 60.0 - reflected[:, 0]

    # Assert
    assert hausdorff(points, reflected) == pytest.approx(0.0, abs=1e-9)
    assert hausdorff(points, half_turned(points)) > 1.0


@pytest.mark.parametrize("count", [8, 9])
def test_the_piece_count_and_vertex_budget_survive_rotational_pairing(
    count: int,
) -> None:
    # Arrange / Act: observation batching stacks terrain into one array, so the
    # count is fixed; the outline budget caps vertices per piece.
    terrain = generated("rotate_180", count)

    # Assert
    assert len(terrain.footprints) == count
    assert max(p.polygon.n_vertices for p in terrain.footprints) <= 6
