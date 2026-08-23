"""The generated tables carry the objectives the published layouts do.

External ground truth, and the only expectation here that does not come from our
own reasoning. Positions were read off the published layout cards -- each card
draws an icon on every objective and states its own total -- and are stored in
`tests/data/published_objective_positions.json` as plain numbers.

This is the check that mattered. Five defects in objective resolution passed
every structural test in this repo before it existed: counts, zone splits,
resolution rates, all-or-nothing coverage, and invariance under simplification.
Each was caught by looking at a picture. A position that is out by twelve inches
is a different ruin, and only a comparison against the real layout says so.

Only geometry is stored. The layouts' own names and vocabulary stay out of the
repo -- see `tests/test_no_ip_references.py`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from scripts.measure_maps import load_maps

SHIPPED_MAPS = Path("configs/evaluation/maps")
PUBLISHED = Path("tests/data/published_objective_positions.json")

# An icon is drawn at its ruin's centre, so it lands within a couple of inches
# of the outline's centroid. Three inches is one control range and is a
# comfortable margin either way; a wrong ruin is out by twelve or more.
TOLERANCE_IN = 3.0


def _centroid(area: list[tuple[float, float]]) -> tuple[float, float]:
    xs = [p[0] for p in area]
    ys = [p[1] for p in area]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def test_every_published_objective_has_one_of_ours_on_it() -> None:
    published = json.loads(PUBLISHED.read_text())
    ours = {m.name: m for m in load_maps(SHIPPED_MAPS)}

    assert set(published) == set(ours)
    for name, positions in published.items():
        mine = [_centroid(o.area) for o in ours[name].objectives or [] if o.area]
        for x, y in positions:
            nearest = min(math.hypot(x - a, y - b) for a, b in mine)
            assert nearest <= TOLERANCE_IN, (
                f"{name}: ({x}, {y}) is {nearest:.1f}in away"
            )


def test_the_split_is_twenty_four_fives_and_twenty_one_sixes() -> None:
    """Matches the published totals, and the hand-traced tables before them.

    A table carries six when a marker is equidistant from two equally large
    ruins and designates both -- the boards are point-symmetric, so the centre
    marker routinely sits between a ruin and its own reflection. The published
    cards draw that as two Centre icons.
    """
    counts = [len(m.objectives or []) for m in load_maps(SHIPPED_MAPS)]

    assert sorted(counts) == [5] * 24 + [6] * 21


def test_the_objectives_are_balanced_across_the_two_deployment_zones() -> None:
    """82 / 82 / 82, and it falls out rather than being aimed at.

    The tables are point-symmetric, so any correct resolution has to put the
    same number of objectives in each side's third of the board. Every wrong
    rule tried here broke this, by two to nine objectives.
    """
    counts = {"player": 0, "middle": 0, "opponent": 0}
    for terrain_map in load_maps(SHIPPED_MAPS):
        for objective in terrain_map.objectives or []:
            assert objective.area is not None
            x, _ = _centroid(objective.area)
            counts["player" if x <= 20 else "opponent" if x >= 40 else "middle"] += 1

    assert counts == {"player": 82, "middle": 82, "opponent": 82}
