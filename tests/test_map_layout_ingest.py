"""The layout API to `TerrainMapConfig` conversion.

Only the pure half is covered: the network fetch is one `urlopen` and mocking it
would test the mock. What matters here is the geometry, because every one of
these functions was written against a convention the API does not document and a
silent error would move terrain rather than crash.
"""

from __future__ import annotations

import math

import pytest
from pydantic_yaml import parse_yaml_raw_as

from scripts.fetch_map_layouts import (
    objectives_for,
    piece_outline,
    render_map_yaml,
    simplify_outline,
)
from wargame_rl.wargame.envs.types import TerrainMapConfig
from wargame_rl.wargame.envs.types.geometry import Polygon


def _circle(cx: float, cy: float, radius: float, n: int) -> list[tuple[float, float]]:
    """A dense ring, standing in for the 167-348 vertex source outlines."""
    return [
        (
            cx + radius * math.cos(2 * math.pi * i / n),
            cy + radius * math.sin(2 * math.pi * i / n),
        )
        for i in range(n)
    ]


def _piece(
    x: float, y: float, rotation: float, points: list[tuple[float, float]]
) -> dict:
    return {
        "footprint": {
            "origin": {"x": x, "y": y},
            "widthIn": 4,
            "heightIn": 2,
            "rotationDeg": rotation,
        },
        "outline": {"points": [{"x": px, "y": py} for px, py in points]},
    }


def test_outline_points_are_centred_on_the_footprint_origin() -> None:
    """The convention the whole ingest rests on, pinned by a hand-checked case."""
    # A 2x2 square about its own centre, placed at board-centre origin (0, 0),
    # which is the middle of the board once shifted.
    square = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]

    outline = piece_outline(_piece(0.0, 0.0, 0.0, square))

    assert outline == [(29.0, 21.0), (31.0, 21.0), (31.0, 23.0), (29.0, 23.0)]


def test_rotation_turns_the_piece_about_its_own_origin() -> None:
    """A 90-degree turn maps (+x) to (+y), around the footprint origin."""
    outline = piece_outline(
        _piece(10.0, 0.0, 90.0, [(2.0, 0.0), (0.0, 1.0), (-2.0, 0.0)])
    )

    assert outline[0] == pytest.approx((40.0, 24.0))


@pytest.mark.parametrize("n_source", [167, 348])
@pytest.mark.parametrize("budget", [4, 8])
def test_simplify_respects_the_budget_and_keeps_the_shape(
    n_source: int, budget: int
) -> None:
    """The observation cannot carry more than the budget, and area must survive."""
    dense = _circle(30.0, 22.0, 5.0, n_source)

    simplified = simplify_outline(dense, budget)

    assert 3 <= len(simplified) <= budget
    ratio = Polygon.from_points(simplified).area / Polygon.from_points(dense).area
    # An inscribed polygon loses area; a 4-gon in a circle keeps 2/pi of it.
    assert 0.63 <= ratio <= 1.0


def test_simplify_keeps_a_concave_bay() -> None:
    """An L-shaped ruin must not come back as its bounding box.

    This is the whole reason the footprint rectangle was rejected: it fills the
    bay in and blocks board the piece does not.
    """
    ell = [(0.0, 0.0), (10.0, 0.0), (10.0, 4.0), (4.0, 4.0), (4.0, 10.0), (0.0, 10.0)]

    simplified = simplify_outline(ell, 8)

    assert Polygon.from_points(simplified).area == pytest.approx(64.0)
    assert not Polygon.from_points(simplified).contains(8.0, 8.0)


def test_a_marker_on_a_ruin_makes_that_ruin_the_objective() -> None:
    pieces = [
        Polygon.from_rect(0.0, 0.0, 4.0, 4.0),
        Polygon.from_rect(20.0, 20.0, 24.0, 24.0),
    ]

    objectives = objectives_for([(22.0, 22.0)], pieces)

    assert objectives == [
        {"area": [[20.0, 20.0], [24.0, 20.0], [24.0, 24.0], [20.0, 24.0]]}
    ]


def test_two_markers_on_one_ruin_are_one_objective() -> None:
    """That ground is held once, so the second marker adds nothing."""
    pieces = [Polygon.from_rect(0.0, 0.0, 10.0, 10.0)]

    objectives = objectives_for([(2.0, 2.0), (8.0, 8.0)], pieces)

    assert len(objectives) == 1


def test_a_marker_on_open_ground_stays_a_disc() -> None:
    """Beyond control range there is no ruin to be the objective."""
    pieces = [Polygon.from_rect(0.0, 0.0, 4.0, 4.0)]

    objectives = objectives_for([(30.0, 22.0)], pieces)

    assert objectives == [{"x": 30.0, "y": 22.0, "radius_size": 3.0}]


def test_a_marker_just_inside_control_range_takes_the_ruin() -> None:
    """The boundary case, since eight real markers sit exactly 3.00in out."""
    pieces = [Polygon.from_rect(0.0, 0.0, 4.0, 4.0)]

    assert "area" in objectives_for([(6.9, 2.0)], pieces)[0]
    assert "x" in objectives_for([(7.1, 2.0)], pieces)[0]


def test_the_rendered_file_parses_as_a_map() -> None:
    """The generator's output has to be loadable by the env, not merely valid YAML."""
    pieces = [
        Polygon.from_rect(1.0, 1.0, 5.0, 5.0),
        Polygon.from_rect(20.0, 20.0, 24.0, 24.0),
    ]
    objectives = objectives_for([(3.0, 3.0), (40.0, 40.0)], pieces)

    parsed = parse_yaml_raw_as(
        TerrainMapConfig, render_map_yaml("table_99", pieces, objectives)
    )

    assert parsed.name == "table_99"
    assert len(parsed.terrain) == 2
    assert parsed.objectives is not None
    assert parsed.objectives[0].area is not None
    assert parsed.objectives[1].x == 40.0
