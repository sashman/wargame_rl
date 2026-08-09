"""Line-of-sight domain: the sampled ray and the blocking composition."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from wargame_rl.wargame.envs.domain.los import segments_are_clear
from wargame_rl.wargame.envs.types import TerrainPieceConfig, WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv

_STEP = 0.25


def _trace(
    segments: list[tuple[float, float, float, float]],
    blockers: list[tuple[float, float, float, float]],
    *,
    exempt: np.ndarray | None = None,
    opaque_cells: np.ndarray | None = None,
) -> np.ndarray:
    """Trace a handful of segments against a handful of rectangles."""
    starts = np.array([[s[0], s[1]] for s in segments], dtype=float)
    ends = np.array([[s[2], s[3]] for s in segments], dtype=float)
    rects = np.array(blockers, dtype=float).reshape(-1, 4)
    return segments_are_clear(
        starts,
        ends,
        rects,
        sample_step=_STEP,
        blocker_exempt=exempt,
        opaque_cells=opaque_cells,
    )


def test_a_zero_length_segment_is_clear_through_solid_matter() -> None:
    """Only the strict interior is sampled, and a point has none.

    A model standing inside a ruin has to be able to see itself, or the see-out
    rule has nothing to build on.
    """
    assert _trace([(3.0, 3.0, 3.0, 3.0)], [(0.0, 0.0, 10.0, 10.0)]).tolist() == [True]


def test_a_rectangle_astride_the_line_blocks_and_one_beside_it_does_not() -> None:
    clear = _trace(
        [(0.0, 5.0, 19.0, 5.0), (0.0, 5.0, 19.0, 5.0)],
        [(8.0, 4.0, 12.0, 6.0), (8.0, 14.0, 12.0, 18.0)],
        exempt=np.array([[False, False], [True, True]]),
    )
    assert clear.tolist() == [False, True]


def test_endpoints_are_never_sampled() -> None:
    """A blocker covering only an endpoint cannot block.

    The see-out / see-into rule is expressed as an *exemption* rather than by
    trimming the ray, so the ray itself must not consult its own endpoints —
    otherwise a model standing on a ruin's edge blocks its own sight.
    """
    # Sitting on the target end and reaching away from the segment: no sample.
    assert _trace([(0.0, 0.0, 2.0, 0.0)], [(2.0, -0.5, 2.5, 0.5)]).tolist() == [True]
    # Sitting on the observer end, likewise.
    assert _trace([(0.0, 0.0, 2.0, 0.0)], [(-0.5, -0.5, 0.0, 0.5)]).tolist() == [True]
    # Reaching back far enough to cover interior ground: now it blocks.
    assert _trace([(0.0, 0.0, 2.0, 0.0)], [(1.5, -0.5, 2.5, 0.5)]).tolist() == [False]


def test_an_exempt_blocker_is_ignored_even_when_the_ray_crosses_it() -> None:
    segments = [(0.0, 5.0, 19.0, 5.0)]
    blockers = [(8.0, 4.0, 12.0, 6.0)]
    assert _trace(segments, blockers).tolist() == [False]
    assert _trace(segments, blockers, exempt=np.array([[True]])).tolist() == [True]


def test_the_sample_step_is_the_resolution_guarantee() -> None:
    """A blocker thinner than the step can be missed. This is why configs bound it.

    Documented as a limit rather than hidden: the failure is silent otherwise,
    and "terrain that does not block" reads as a terrain-tuning problem rather
    than a sampling one.
    """
    fat = _trace([(0.0, 0.5, 10.0, 0.5)], [(5.0, 0.0, 5.0 + _STEP, 1.0)])
    thin = _trace([(0.0, 0.5, 10.0, 0.5)], [(5.001, 0.0, 5.002, 1.0)])
    assert fat.tolist() == [False]
    assert thin.tolist() == [True]


def test_the_opaque_cell_grid_blocks_by_cell_index() -> None:
    """The legacy `blocking_mask` is a raster and stays one, indexed [y][x]."""
    cells = np.zeros((10, 10), dtype=bool)
    cells[0, 3] = True
    clear = _trace([(0.0, 0.5, 6.0, 0.5), (0.0, 5.5, 6.0, 5.5)], [], opaque_cells=cells)
    assert clear.tolist() == [False, True]


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
    assert env.has_line_of_sight_between_points(0, 0, 2, 2) is False
    assert env.has_line_of_sight_between_points(0, 0, 2, 0) is True


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


# --- Terrain LOS behavioural tests ---


def _terrain_env(
    terrain: list[TerrainPieceConfig],
    *,
    board_width: int = 20,
    board_height: int = 20,
    blocking_mask: list[list[bool]] | None = None,
) -> WargameEnv:
    """Helper to build a minimal env with terrain footprints."""
    cfg = WargameEnvConfig(
        board_width=board_width,
        board_height=board_height,
        terrain=terrain,
        blocking_mask=blocking_mask,
        number_of_wargame_models=1,
        number_of_objectives=1,
        render_mode=None,
        number_of_battle_rounds=1,
    )
    return WargameEnv(cfg)


def test_terrain_los_blocked_between_outside_models() -> None:
    """Both models outside footprint between them -> LOS blocked."""
    # Footprint at columns 8-12 on row 5; observer (0,5), target (19,5)
    env = _terrain_env([TerrainPieceConfig(footprint=(8, 4, 12, 6))])
    assert env.has_line_of_sight_between_points(0, 5, 19, 5) is False


def test_terrain_los_see_into_target_inside() -> None:
    """Target inside footprint -> LOS clear (see-into rule)."""
    env = _terrain_env([TerrainPieceConfig(footprint=(8, 4, 12, 6))])
    assert env.has_line_of_sight_between_points(0, 5, 10, 5) is True


def test_terrain_los_see_out_observer_inside() -> None:
    """Observer inside footprint -> LOS clear (see-out rule)."""
    env = _terrain_env([TerrainPieceConfig(footprint=(8, 4, 12, 6))])
    assert env.has_line_of_sight_between_points(10, 5, 19, 5) is True


def test_terrain_los_per_ruin_other_ruin_still_blocks() -> None:
    """Inside ruin A, ruin B between observer and target -> blocked by ruin B."""
    env = _terrain_env(
        [
            TerrainPieceConfig(footprint=(0, 4, 3, 6)),  # ruin A (observer inside)
            TerrainPieceConfig(footprint=(8, 4, 12, 6)),  # ruin B (between)
        ]
    )
    assert env.has_line_of_sight_between_points(2, 5, 19, 5) is False


def test_terrain_los_off_line_footprint_unaffected() -> None:
    """Footprint not on the LOS line -> LOS clear."""
    env = _terrain_env([TerrainPieceConfig(footprint=(8, 14, 12, 18))])
    assert env.has_line_of_sight_between_points(0, 5, 19, 5) is True


def test_terrain_los_interior_only_endpoint_footprint_does_not_block() -> None:
    """A footprint cell coinciding with an endpoint doesn't block (endpoint excluded)."""
    # Footprint at (5,5)-(5,5), target at (5,5)
    env = _terrain_env([TerrainPieceConfig(footprint=(5, 5, 5, 5))])
    assert env.has_line_of_sight_between_points(0, 5, 5, 5) is True


def test_terrain_los_blocking_mask_and_footprint_coexist() -> None:
    """Both blocking_mask and footprint configured; either one blocks via OR."""
    mask = [[False] * 20 for _ in range(20)]
    mask[5][15] = True  # block cell (15, 5)
    env = _terrain_env(
        [TerrainPieceConfig(footprint=(8, 4, 12, 6))],
        blocking_mask=mask,
    )
    # Blocked by footprint
    assert env.has_line_of_sight_between_points(0, 5, 19, 5) is False
    # Blocked by mask cell (15, 5) — line from (13, 5) to (19, 5) passes through (15, 5)
    assert env.has_line_of_sight_between_points(13, 5, 19, 5) is False
    # Clear line that avoids both
    assert env.has_line_of_sight_between_points(0, 0, 5, 0) is True


@given(
    board_w=st.integers(min_value=8, max_value=20),
    board_h=st.integers(min_value=8, max_value=20),
    fp_data=st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=15),
            st.integers(min_value=0, max_value=15),
            st.integers(min_value=1, max_value=3),
            st.integers(min_value=1, max_value=3),
        ),
        min_size=0,
        max_size=2,
    ),
    x0=st.integers(min_value=0, max_value=19),
    y0=st.integers(min_value=0, max_value=19),
    x1=st.integers(min_value=0, max_value=19),
    y1=st.integers(min_value=0, max_value=19),
)
@settings(max_examples=200, deadline=None)
def test_terrain_los_symmetry(
    board_w: int,
    board_h: int,
    fp_data: list[tuple[int, int, int, int]],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> None:
    """has_los(A,B) == has_los(B,A) over random boards/footprints/endpoints.

    Symmetry is not decorative: `firepower_ratio` reads an exposed model as one
    that can also fire, so an asymmetric trace would make that metric count
    different populations in each direction.
    """
    assume(x0 < board_w and y0 < board_h and x1 < board_w and y1 < board_h)

    footprints: list[TerrainPieceConfig] = []
    for fx, fy, fw, fh in fp_data:
        fx1 = fx + fw
        fy1 = fy + fh
        assume(fx1 < board_w and fy1 < board_h)
        # Reject if overlaps any previous footprint
        for prev in footprints:
            px0, py0, px1, py1 = prev.footprint
            if fx <= px1 and fx1 >= px0 and fy <= py1 and fy1 >= py0:
                assume(False)
        footprints.append(TerrainPieceConfig(footprint=(fx, fy, fx1, fy1)))

    env = _terrain_env(footprints, board_width=board_w, board_height=board_h)
    forward = env.has_line_of_sight_between_points(x0, y0, x1, y1)
    backward = env.has_line_of_sight_between_points(x1, y1, x0, y0)
    assert forward == backward
