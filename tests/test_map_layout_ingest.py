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
    merge_ruin,
    objectives_for,
    piece_outline,
    render_map_yaml,
    ruin_components,
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


def test_a_marker_on_open_ground_still_takes_a_ruin() -> None:
    """There is no such thing as an objective that is not a ruin.

    Eight of the pool's 225 markers sit beyond control range of every piece --
    at most 5.16in, four of them the exact board centre. They resolve to the
    nearest ruin anyway: a disc floating on open ground would be the previous
    edition's free-standing marker under a new name. Six of the eight are at
    exactly 3.00in, one control range, so that ruin is precisely the ground you
    would hold the marker from.
    """
    pieces = [Polygon.from_rect(0.0, 0.0, 4.0, 4.0)]

    objectives = objectives_for([(30.0, 22.0)], pieces)

    assert len(objectives) == 1
    assert "area" in objectives[0]
    assert "radius_size" not in objectives[0]


def test_the_nearest_ruin_wins_however_far_away_it_is() -> None:
    near = Polygon.from_rect(0.0, 0.0, 4.0, 4.0)
    far = Polygon.from_rect(50.0, 40.0, 54.0, 43.0)

    objectives = objectives_for([(9.0, 2.0)], [near, far])

    area = Polygon.from_points([(x, y) for x, y in objectives[0]["area"]])
    assert area.contains(2.0, 2.0), "the near ruin, not the far one"


def test_the_rendered_file_parses_as_a_map() -> None:
    """The generator's output has to be loadable by the env, not merely valid YAML."""
    pieces = [
        Polygon.from_rect(1.0, 1.0, 5.0, 5.0),
        Polygon.from_rect(20.0, 20.0, 24.0, 24.0),
    ]
    objectives = objectives_for([(3.0, 3.0), (22.0, 22.0)], pieces)
    deployment = {
        "name": "short_edges",
        "player": [(0.0, 0.0), (18.0, 0.0), (18.0, 44.0), (0.0, 44.0)],
        "opponent": [(42.0, 0.0), (60.0, 0.0), (60.0, 44.0), (42.0, 44.0)],
    }

    parsed = parse_yaml_raw_as(
        TerrainMapConfig,
        render_map_yaml("table_99", pieces, objectives, deployment),
    )

    assert parsed.name == "table_99"
    assert len(parsed.terrain) == 2
    assert parsed.objectives is not None
    assert len(parsed.objectives) == 2
    assert parsed.deployment is not None
    assert parsed.deployment.name == "short_edges"
    assert parsed.deployment.player_polygon().area == 18.0 * 44.0
    assert all(o.area is not None for o in parsed.objectives)
    assert parsed.deployment is not None
    assert parsed.deployment.name == "short_edges"
    assert parsed.deployment.player_polygon().area == 18.0 * 44.0


def test_a_diagonal_seam_is_one_ruin() -> None:
    """The failure that shipped: a rectangle split in two along a diagonal.

    On `table_02` the centre ruin is exactly this, and resolving the marker to
    the nearest *piece* made the objective cover one side of the seam only.
    """
    lower = Polygon.from_points([(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)])
    upper = Polygon.from_points([(10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])

    components = ruin_components([lower, upper])

    assert [sorted(group) for group in components] == [[0, 1]]


def test_two_bars_butted_into_an_l_are_one_ruin() -> None:
    """The other real shape: a bar's end against another bar's side."""
    upright = Polygon.from_points([(0.0, 0.0), (2.0, 0.0), (2.0, 8.0), (0.0, 8.0)])
    across = Polygon.from_points([(2.0, 0.0), (8.0, 0.0), (8.0, 2.0), (2.0, 2.0)])

    assert [sorted(g) for g in ruin_components([upright, across])] == [[0, 1]]


def test_a_corner_touch_is_two_ruins() -> None:
    """Incidental contact must not merge — 110 of 222 real pairs are this."""
    first = Polygon.from_points([(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)])
    second = Polygon.from_points([(4.0, 4.0), (8.0, 4.0), (8.0, 8.0), (4.0, 8.0)])

    assert [sorted(g) for g in ruin_components([first, second])] == [[0], [1]]


def test_merging_a_ruin_keeps_all_of_its_ground() -> None:
    """The union must be one outline covering both pieces, not the larger one.

    Two abutting polygons do not fuse on their own — shapely returns a
    MultiPolygon — and the first implementation took the biggest part, which
    silently threw half the ruin away while every count still read five.
    """
    upright = Polygon.from_points([(0.0, 0.0), (2.0, 0.0), (2.0, 8.0), (0.0, 8.0)])
    across = Polygon.from_points([(2.0, 0.0), (8.0, 0.0), (8.0, 2.0), (2.0, 2.0)])

    merged = Polygon.from_points(merge_ruin([upright, across], [0, 1]))

    assert merged.area == pytest.approx(16.0 + 12.0, abs=0.5)
    assert merged.contains(1.0, 7.0)  # the top of the upright
    assert merged.contains(7.0, 1.0)  # the far end of the crossbar
    assert not merged.contains(7.0, 7.0)  # the notch stays open


def test_a_marker_takes_the_whole_ruin_not_the_piece_it_landed_on() -> None:
    upright = Polygon.from_points([(0.0, 0.0), (2.0, 0.0), (2.0, 8.0), (0.0, 8.0)])
    across = Polygon.from_points([(2.0, 0.0), (8.0, 0.0), (8.0, 2.0), (2.0, 2.0)])

    objectives = objectives_for([(1.0, 7.0)], [upright, across])

    assert len(objectives) == 1
    area = Polygon.from_points([(x, y) for x, y in objectives[0]["area"]])
    assert area.contains(7.0, 1.0), "the crossbar is part of the same ruin"


def test_a_ruin_merges_across_the_rounding_gap() -> None:
    """The real pieces do not share an exact edge, and that is the whole problem.

    Map files carry two decimal places and the outlines are simplified, so two
    abutting pieces sit a few hundredths apart. `unary_union` leaves those as a
    MultiPolygon, which is why the union is bridged. Fixtures that share an
    exact edge fuse on their own and cannot catch a missing bridge -- an
    injected "take the largest part" passed every other test here.
    """
    upright = Polygon.from_points([(0.0, 0.0), (2.0, 0.0), (2.0, 8.0), (0.0, 8.0)])
    across = Polygon.from_points([(2.05, 0.0), (8.0, 0.0), (8.0, 2.0), (2.05, 2.0)])

    assert [sorted(g) for g in ruin_components([upright, across])] == [[0, 1]]

    merged = Polygon.from_points(merge_ruin([upright, across], [0, 1]))

    assert merged.contains(1.0, 7.0), "the upright"
    assert merged.contains(7.0, 1.0), "the crossbar, across the gap"
    assert merged.area == pytest.approx(16.0 + 11.9, abs=0.6)


def test_each_marker_gets_its_own_ruin() -> None:
    """Two markers never collapse onto one objective.

    Merging them was the first rule here, on the reasoning that one piece of
    ground is held once. The pool refuted it: the only collisions were markers
    **twelve to seventeen inches apart** both reaching one long ruin, which are
    plainly two objectives. Each marker now takes the largest *unclaimed* ruin
    in range.
    """
    big = Polygon.from_points([(0.0, 0.0), (30.0, 0.0), (30.0, 6.0), (0.0, 6.0)])
    small_a = Polygon.from_points([(0.0, 8.0), (4.0, 8.0), (4.0, 12.0), (0.0, 12.0)])
    small_b = Polygon.from_points(
        [(26.0, 8.0), (30.0, 8.0), (30.0, 12.0), (26.0, 12.0)]
    )

    objectives = objectives_for([(2.0, 7.0), (28.0, 7.0)], [big, small_a, small_b])

    assert len(objectives) == 2
    areas = {
        Polygon.from_points([(x, y) for x, y in o["area"]]).area for o in objectives
    }
    assert len(areas) == 2, "the two markers took different ruins"


def test_the_biggest_ruin_in_range_wins_not_the_nearest() -> None:
    """The defect that shipped on table_01 and table_10.

    A marker sitting in the gap between a scrap of scatter terrain and a real
    ruin handed the objective to the scrap, because the scrap was an inch
    closer. Control range is the rule's own measure: every ruin within 3in is
    ground you could hold the marker from, so the one that dominates wins.
    """
    scrap = Polygon.from_points([(0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (0.0, 3.0)])
    ruin = Polygon.from_points([(6.0, 0.0), (18.0, 0.0), (18.0, 8.0), (6.0, 8.0)])

    # 1in from the scrap, 2in from the ruin -- nearest would take the scrap.
    objectives = objectives_for([(4.0, 1.5)], [scrap, ruin])

    area = Polygon.from_points([(x, y) for x, y in objectives[0]["area"]])
    assert area.contains(12.0, 4.0), "the ruin, not the scrap"


def test_a_marker_with_nothing_in_range_still_takes_the_nearest() -> None:
    near = Polygon.from_rect(0.0, 0.0, 4.0, 4.0)
    far = Polygon.from_rect(50.0, 40.0, 54.0, 43.0)

    objectives = objectives_for([(20.0, 20.0)], [near, far])

    area = Polygon.from_points([(x, y) for x, y in objectives[0]["area"]])
    assert area.contains(2.0, 2.0)


def test_a_marker_between_two_equal_ruins_designates_both() -> None:
    """The board centre sits in the gap between a ruin and its own reflection.

    These tables are point-symmetric, so an equidistant pair of equally large
    ruins is the normal case at the centre, not a curiosity. On `table_01` the
    centre marker is 1.02in from each of two 58.5 sq in wedges whose centroids
    are exact mirrors; taking one by list order dropped an objective the layout
    intends and broke the table's symmetry.
    """
    left = Polygon.from_points([(0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0)])
    right = Polygon.from_points([(12.0, 0.0), (20.0, 0.0), (20.0, 8.0), (12.0, 8.0)])

    objectives = objectives_for([(10.0, 4.0)], [left, right])

    assert len(objectives) == 2, "both halves of a symmetric pair"


def test_an_unequal_pair_is_not_a_tie() -> None:
    """Only a genuine tie splits; a bigger ruin still wins outright."""
    big = Polygon.from_points([(0.0, 0.0), (16.0, 0.0), (16.0, 8.0), (0.0, 8.0)])
    small = Polygon.from_points([(18.0, 3.0), (21.0, 3.0), (21.0, 6.0), (18.0, 6.0)])

    objectives = objectives_for([(17.0, 4.0)], [big, small])

    assert len(objectives) == 1
    area = Polygon.from_points([(x, y) for x, y in objectives[0]["area"]])
    assert area.contains(8.0, 4.0), "the big ruin"


def test_a_big_ruin_just_out_of_control_range_still_wins() -> None:
    """`MARKER_REACH_IN` is an authoring distance, not the rules' control range.

    The layouts routinely place a marker 3.75-4.0in from the large ruin it
    plainly designates, with a scrap of scatter terrain 2-3in away on the other
    side. Cutting candidates at the rules' 3in handed the objective to the
    scrap on `table_02` and `table_06`; both then disagreed with every one of
    the hand-traced objectives for those tables.
    """
    scrap = Polygon.from_points([(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)])
    ruin = Polygon.from_points([(10.0, 0.0), (22.0, 0.0), (22.0, 8.0), (10.0, 8.0)])

    # 2in from the scrap, 4in from the ruin -- outside control range, inside reach.
    objectives = objectives_for([(6.0, 2.0)], [scrap, ruin])

    assert len(objectives) == 1
    area = Polygon.from_points([(x, y) for x, y in objectives[0]["area"]])
    assert area.contains(16.0, 4.0), "the ruin four inches away, not the scrap two away"
