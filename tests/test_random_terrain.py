"""Tests for per-episode random terrain generation.

Randomising terrain is what makes a cover result falsifiable — with a fixed
layout, "the policy used cover" and "the policy memorised seven rectangles"
produce identical numbers. These tests pin the invariants that has to hold to:
the layout really does change, it stays legal, and the piece count never varies,
because observation batching stacks terrain into a single array.
"""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.envs.domain.terrain_placement import generate_terrain
from wargame_rl.wargame.envs.domain.value_objects import BoardDimensions
from wargame_rl.wargame.envs.types import (
    ModelConfig,
    RandomTerrainConfig,
    TerrainPieceConfig,
    WargameEnvConfig,
)
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.observation import observations_to_tensor_batch

BOARD = BoardDimensions(width=60, height=44)

Rect = tuple[int, int, int, int]


def _rects(terrain: object) -> list[Rect]:
    """Footprints as sorted corner tuples, for order-insensitive comparison."""
    return sorted(
        (f.x0, f.y0, f.x1, f.y1)
        for f in terrain.footprints  # type: ignore[attr-defined]
    )


def _layout(seed: int, **overrides: object) -> list[Rect]:
    spec = RandomTerrainConfig(**overrides)  # type: ignore[arg-type]
    return _rects(generate_terrain(spec, BOARD, np.random.default_rng(seed)))


def _overlaps(a: Rect, b: Rect) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def test_layout_is_seeded_and_varies() -> None:
    """Same seed reproduces a layout; different seeds give different ones."""
    assert _layout(7) == _layout(7)
    assert _layout(1) != _layout(2)


@pytest.mark.parametrize("seed", range(8))
def test_layout_is_legal(seed: int) -> None:
    """Generated footprints satisfy the same rules a hand-written list must.

    `validate_terrain` enforces in-bounds and non-overlapping for configured
    terrain; generated terrain bypasses that validator, so the guarantee has to
    be tested at the source.
    """
    rects = _layout(seed, edge_margin=2)

    for x0, y0, x1, y1 in rects:
        assert 2 <= x0 <= x1 < BOARD.width - 2
        assert 2 <= y0 <= y1 < BOARD.height - 2

    for i, a in enumerate(rects):
        for b in rects[i + 1 :]:
            assert not _overlaps(a, b), f"{a} overlaps {b}"


@pytest.mark.parametrize("seed", range(8))
def test_mirrored_layout_is_symmetric(seed: int) -> None:
    """Both deployment zones get the same ground.

    Zones are fixed to the left and right of the board and `turn_order` only
    swaps who moves first, so an asymmetric draw would favour one side for a
    whole run.
    """
    rects = _layout(seed, mirror=True)
    reflected = sorted(
        (BOARD.width - 1 - x1, y0, BOARD.width - 1 - x0, y1) for x0, y0, x1, y1 in rects
    )
    assert reflected == rects


@pytest.mark.parametrize("count,mirror", [(7, True), (6, True), (7, False), (1, True)])
def test_piece_count_is_exact(count: int, mirror: bool) -> None:
    """The count is the contract that lets observations batch — odd or even."""
    for seed in range(5):
        assert len(_layout(seed, count=count, mirror=mirror)) == count


def test_unsatisfiable_spec_is_rejected_at_config_load() -> None:
    """An over-packed board fails at load, not deep inside a training run."""
    with pytest.raises(ValueError, match="packs too tightly"):
        WargameEnvConfig(
            board_width=30,
            board_height=30,
            random_terrain=RandomTerrainConfig(count=20, min_size=7, max_size=7),
        )


def test_fixed_and_random_terrain_are_mutually_exclusive() -> None:
    """Terrain is either fixed or regenerated — silently ignoring one is worse."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        WargameEnvConfig(
            board_width=60,
            board_height=44,
            terrain=[TerrainPieceConfig(footprint=(1, 1, 5, 5))],
            random_terrain=RandomTerrainConfig(),
        )


def _env(random_terrain: RandomTerrainConfig | None) -> WargameEnv:
    config = WargameEnvConfig(
        board_width=40,
        board_height=30,
        number_of_wargame_models=3,
        number_of_objectives=2,
        models=[ModelConfig() for _ in range(3)],
        random_terrain=random_terrain,
    )
    return WargameEnv(config, renderer=None)


def test_env_regenerates_terrain_every_episode() -> None:
    """Reset draws a new layout, and the same seed draws the same one."""
    env = _env(RandomTerrainConfig(count=5, min_size=4, max_size=6))
    try:
        env.reset(seed=1)
        first = _rects(env.terrain)
        env.reset(seed=2)
        second = _rects(env.terrain)
        env.reset(seed=1)
        again = _rects(env.terrain)
    finally:
        env.close()

    assert first != second
    assert first == again


def test_observations_from_different_episodes_batch() -> None:
    """The regression the fixed piece count exists to prevent.

    `observations_to_tensor_batch` stacks terrain into one array, so episodes
    with different piece counts would fail to collate — mid-training, on a
    batch boundary, far from the config that caused it.
    """
    env = _env(RandomTerrainConfig(count=5, min_size=4, max_size=6))
    try:
        observations = [env.reset(seed=seed)[0] for seed in range(4)]
    finally:
        env.close()

    tensors = observations_to_tensor_batch(observations)
    assert tuple(tensors[4].shape) == (4, 5, 4)


def test_terrain_is_stable_without_random_terrain() -> None:
    """Backward compatibility: configs that never opted in are unaffected."""
    env = _env(None)
    try:
        env.reset(seed=1)
        first = _rects(env.terrain)
        env.reset(seed=99)
        second = _rects(env.terrain)
    finally:
        env.close()

    assert first == second
