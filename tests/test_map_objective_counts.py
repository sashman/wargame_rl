"""The generated tables carry the objective counts the published layouts do.

External ground truth, and the only check here that does not come from our own
reasoning. The counts were read off the published layout cards for all 45
layouts -- each states its own objective total -- and they agree exactly with
the counts the hand-traced tables carried before they were generated.

That matters because five of the six defects found while building this ingest
passed every structural check and had to be caught by eye. A count that is
wrong by one is the visible symptom of every one of them: an objective covering
half a ruin, resolving to a scrap of scatter terrain, collapsing onto a
neighbour, or a symmetric pair silently reduced to one.

Only counts are recorded. The layouts' own names and vocabulary stay out of the
repo -- see `tests/test_no_ip_references.py`.
"""

from __future__ import annotations

from pathlib import Path

from scripts.measure_maps import load_maps

SHIPPED_MAPS = Path("configs/evaluation/maps")

PUBLISHED_OBJECTIVE_COUNTS = {
    "table_01": 6,
    "table_02": 5,
    "table_03": 6,
    "table_04": 5,
    "table_05": 6,
    "table_06": 5,
    "table_07": 5,
    "table_08": 5,
    "table_09": 5,
    "table_10": 5,
    "table_11": 6,
    "table_12": 5,
    "table_13": 5,
    "table_14": 5,
    "table_15": 6,
    "table_16": 6,
    "table_17": 5,
    "table_18": 5,
    "table_19": 6,
    "table_20": 6,
    "table_21": 6,
    "table_22": 5,
    "table_23": 6,
    "table_24": 5,
    "table_25": 5,
    "table_26": 6,
    "table_27": 5,
    "table_28": 5,
    "table_29": 6,
    "table_30": 6,
    "table_31": 6,
    "table_32": 5,
    "table_33": 6,
    "table_34": 5,
    "table_35": 6,
    "table_36": 5,
    "table_37": 6,
    "table_38": 5,
    "table_39": 6,
    "table_40": 5,
    "table_41": 5,
    "table_42": 6,
    "table_43": 5,
    "table_44": 6,
    "table_45": 6,
}


def test_every_table_carries_the_published_objective_count() -> None:
    counts = {
        terrain_map.name: len(terrain_map.objectives or [])
        for terrain_map in load_maps(SHIPPED_MAPS)
    }

    assert counts == PUBLISHED_OBJECTIVE_COUNTS


def test_the_split_is_twenty_four_fives_and_twenty_one_sixes() -> None:
    """Pinned separately so a wholesale regeneration cannot drift the mix.

    A table carries six when one of its markers is equidistant from two equally
    large ruins and designates both -- the boards are point-symmetric, so the
    centre marker routinely sits between a ruin and its own reflection.
    """
    counts = [len(m.objectives or []) for m in load_maps(SHIPPED_MAPS)]

    assert sorted(counts) == [5] * 24 + [6] * 21
