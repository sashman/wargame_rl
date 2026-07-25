"""Line-of-sight domain: Bresenham trace and blocking predicate."""

from __future__ import annotations

import pytest

from wargame_rl.wargame.envs.domain.los import has_line_of_sight, iter_los_cells
from wargame_rl.wargame.envs.types import TerrainPieceConfig, WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv


def test_los_same_cell_true_even_if_blocking_everywhere() -> None:
    """Interior is empty; is_blocking must not be consulted for the lone cell."""

    def always_block(_x: int, _y: int) -> bool:
        return True

    assert has_line_of_sight(3, 3, 3, 3, 10, 10, always_block) is True


def test_los_clear_horizontal() -> None:
    def never_block(_x: int, _y: int) -> bool:
        return False

    assert has_line_of_sight(0, 0, 5, 0, 10, 10, never_block) is True


def test_los_blocked_mid_horizontal() -> None:
    blocked = {(3, 0)}

    def is_b(x: int, y: int) -> bool:
        return (x, y) in blocked

    assert has_line_of_sight(0, 0, 5, 0, 10, 10, is_b) is False


def test_los_diagonal_clear_and_blocked() -> None:
    def never_block(_x: int, _y: int) -> bool:
        return False

    assert has_line_of_sight(0, 0, 4, 4, 10, 10, never_block) is True

    def block_center(x: int, y: int) -> bool:
        return (x, y) == (2, 2)

    assert has_line_of_sight(0, 0, 4, 4, 10, 10, block_center) is False


def test_los_out_of_bounds_returns_false_and_empty_iter() -> None:
    """OOB endpoints: no trace, no LOS."""
    assert iter_los_cells(-1, 0, 5, 0, 10, 10) == []

    def never_block_oob(_x: int, _y: int) -> bool:
        return False

    assert has_line_of_sight(-1, 0, 5, 0, 10, 10, never_block_oob) is False
    assert iter_los_cells(0, 0, 5, 10, 10, 10) == []
    assert has_line_of_sight(0, 0, 5, 10, 10, 10, never_block_oob) is False


def test_los_iter_consistency_manual_interior_scan() -> None:
    cells = iter_los_cells(0, 0, 5, 0, 10, 10)
    assert len(cells) >= 2
    interior = cells[1:-1]
    blocked: set[tuple[int, int]] = set()

    def is_b(x: int, y: int) -> bool:
        return (x, y) in blocked

    manual = not any(is_b(x, y) for x, y in interior)
    assert has_line_of_sight(0, 0, 5, 0, 10, 10, is_b) == manual


def test_los_golden_trace_zero_three_one() -> None:
    assert iter_los_cells(0, 0, 3, 1, 10, 10) == [
        (0, 0),
        (1, 0),
        (2, 1),
        (3, 1),
    ]


def test_los_interior_only_blocking_ignores_endpoint_blocker() -> None:
    """Blocker on target cell is not in cells[1:-1] for a 3-cell horizontal line."""
    # (0,0) -> (2,0): interior is [(1, 0)] only; blocking (2,0) must not block LOS.

    def block_target(x: int, y: int) -> bool:
        return (x, y) == (2, 0)

    assert has_line_of_sight(0, 0, 2, 0, 10, 10, block_target) is True


def test_blocking_mask_yaml_default_none_on_fixture_config() -> None:
    """Existing configs without blocking_mask stay None."""
    cfg = WargameEnvConfig(render_mode=None, number_of_battle_rounds=100)
    assert cfg.blocking_mask is None


def test_wargame_env_los_uses_config_mask() -> None:
    # 3x3 board, block center (1,1)
    mask = [
        [False, False, False],
        [False, True, False],
        [False, False, False],
    ]
    cfg = WargameEnvConfig(
        board_width=3,
        board_height=3,
        blocking_mask=mask,
        number_of_wargame_models=1,
        number_of_objectives=1,
        render_mode=None,
        number_of_battle_rounds=1,
    )
    env = WargameEnv(cfg)
    # opposite corners, line passes through (1,1)
    assert env.has_line_of_sight_between_cells(0, 0, 2, 2) is False
    assert env.has_line_of_sight_between_cells(0, 0, 2, 0) is True


def test_iter_los_cells_between_cells_matches_domain() -> None:
    cfg = WargameEnvConfig(
        board_width=5,
        board_height=5,
        number_of_wargame_models=1,
        number_of_objectives=1,
        render_mode=None,
        number_of_battle_rounds=1,
    )
    env = WargameEnv(cfg)
    assert env.iter_los_cells_between_cells(0, 0, 2, 1) == iter_los_cells(
        0, 0, 2, 1, 5, 5
    )


def test_blocking_mask_invalid_shape_raises() -> None:
    with pytest.raises(ValueError, match="board_height"):
        WargameEnvConfig(
            board_width=2,
            board_height=3,
            blocking_mask=[[False, False]],  # wrong row count
            number_of_wargame_models=1,
            number_of_objectives=1,
            render_mode=None,
            number_of_battle_rounds=1,
        )


# --- Terrain config tests ---


def test_terrain_config_parses_footprint() -> None:
    """Valid terrain constructs and stores footprint tuple."""
    cfg = WargameEnvConfig(
        board_width=50,
        board_height=50,
        terrain=[TerrainPieceConfig(footprint=(27, 8, 33, 16))],
        number_of_wargame_models=1,
        number_of_objectives=1,
        render_mode=None,
        number_of_battle_rounds=1,
    )
    assert cfg.terrain is not None
    assert cfg.terrain[0].footprint == (27, 8, 33, 16)


def test_terrain_config_default_none() -> None:
    """No terrain key gives None."""
    cfg = WargameEnvConfig(
        number_of_wargame_models=1,
        number_of_objectives=1,
        render_mode=None,
        number_of_battle_rounds=1,
    )
    assert cfg.terrain is None


def test_terrain_validation_off_board_corner_raises() -> None:
    """Terrain extending beyond board boundary raises ValueError."""
    with pytest.raises(ValueError, match="outside the board"):
        WargameEnvConfig(
            board_width=10,
            board_height=10,
            terrain=[TerrainPieceConfig(footprint=(8, 8, 11, 9))],
            number_of_wargame_models=1,
            number_of_objectives=1,
            render_mode=None,
            number_of_battle_rounds=1,
        )


def test_terrain_validation_overlapping_footprints_raises() -> None:
    """Two overlapping terrain pieces raise ValueError."""
    with pytest.raises(ValueError, match="overlap"):
        WargameEnvConfig(
            board_width=50,
            board_height=50,
            terrain=[
                TerrainPieceConfig(footprint=(5, 5, 10, 10)),
                TerrainPieceConfig(footprint=(8, 8, 15, 15)),
            ],
            number_of_wargame_models=1,
            number_of_objectives=1,
            render_mode=None,
            number_of_battle_rounds=1,
        )


def test_terrain_validation_overlap_with_zone_or_objective_allowed() -> None:
    """Terrain overlapping deployment zones or objective positions is allowed."""
    cfg = WargameEnvConfig(
        board_width=50,
        board_height=50,
        terrain=[TerrainPieceConfig(footprint=(0, 0, 5, 5))],
        deployment_zone=(0, 0, 10, 50),
        number_of_wargame_models=1,
        number_of_objectives=1,
        render_mode=None,
        number_of_battle_rounds=1,
    )
    assert cfg.terrain is not None
