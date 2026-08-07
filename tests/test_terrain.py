"""Unit tests for the pure-domain terrain model (Footprint + Terrain)."""

from __future__ import annotations

from wargame_rl.wargame.envs.domain.terrain import Footprint, Terrain


def test_footprint_contains_inclusive_of_corners() -> None:
    """Corners and interior cells are contained; cells outside are not."""
    fp = Footprint.from_corners(0, 0, 2, 2)
    assert fp.contains(0, 0) is True
    assert fp.contains(2, 2) is True
    assert fp.contains(1, 1) is True
    assert fp.contains(3, 0) is False
    assert fp.contains(0, 3) is False


def test_footprint_normalises_unordered_corners() -> None:
    """from_corners swaps so x0<=x1 and y0<=y1."""
    fp = Footprint.from_corners(2, 2, 0, 0)
    assert fp.x0 == 0
    assert fp.y0 == 0
    assert fp.x1 == 2
    assert fp.y1 == 2
    assert fp.contains(1, 1) is True
    assert fp.contains(3, 3) is False


def test_blocking_footprints_for_endpoints_excludes_footprint_containing_endpoint() -> (
    None
):
    """Footprint containing an endpoint is excluded from blocking candidates."""
    fp = Footprint.from_corners(5, 5, 10, 10)
    terrain = Terrain([fp])

    # Observer inside footprint → excluded
    assert terrain.blocking_footprints_for_endpoints(6, 6, 20, 20) == []
    # Target inside footprint → excluded
    assert terrain.blocking_footprints_for_endpoints(0, 0, 7, 7) == []
    # Both outside footprint → included
    assert terrain.blocking_footprints_for_endpoints(0, 0, 20, 20) == [fp]


def test_terrain_empty_returns_no_blockers() -> None:
    """Empty terrain never blocks."""
    terrain = Terrain([])
    assert terrain.blocking_footprints_for_endpoints(0, 0, 10, 10) == []
