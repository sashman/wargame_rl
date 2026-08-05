"""Tests for the line-of-sight exposure and terrain-proximity metrics.

`exposure_rate` is the measure the cover experiment turns on, so its geometry
has to be exactly right: it must fall when terrain breaks the sightline and only
then. These tests fix the board so the answer is checkable by hand — models are
placed at known cells, the objective sits on the opponents so nothing moves, and
the player is ordered to stay put.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from wargame_rl.wargame.envs.domain.terrain import Footprint
from wargame_rl.wargame.envs.env_components.exposure import (
    ExposureTracker,
    distances_to_nearest_footprint,
)
from wargame_rl.wargame.envs.types import (
    ModelConfig,
    ObjectiveConfig,
    OpponentPolicyConfig,
    TerrainPieceConfig,
    WargameEnvAction,
    WargameEnvConfig,
)
from wargame_rl.wargame.envs.types.config import WeaponProfile
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv

PLAYER_X = 10
OPPONENT_X = 18
N_MODELS = 2

# A ruin spanning the full height between the two lines, so every sightline
# from an opponent to a player crosses it.
BLOCKING_RUIN = TerrainPieceConfig(footprint=(14, 0, 15, 39))


def _make_env(
    *,
    weapon_range: int = 12,
    terrain: list[TerrainPieceConfig] | None = None,
    track_exposure: bool = True,
) -> WargameEnv:
    """Two player models facing two armed opponents across open ground.

    The objective sits on the opponents so `scripted_advance_and_shoot` keeps
    them still, and the player is given `STAY` every step, which holds the whole
    geometry fixed for the episode.
    """
    config = WargameEnvConfig(
        board_width=40,
        board_height=40,
        number_of_wargame_models=N_MODELS,
        number_of_opponent_models=N_MODELS,
        number_of_objectives=1,
        objective_radius_size=3,
        number_of_battle_rounds=3,
        skip_phases=[BattlePhase.command, BattlePhase.charge, BattlePhase.fight],
        # High wounds so nobody dies and the alive-model denominator is constant.
        models=[
            ModelConfig(x=PLAYER_X, y=18 + i, max_wounds=50) for i in range(N_MODELS)
        ],
        opponent_models=[
            ModelConfig(
                x=OPPONENT_X,
                y=18 + i,
                weapons=[WeaponProfile(range=weapon_range)],
            )
            for i in range(N_MODELS)
        ],
        objectives=[ObjectiveConfig(x=OPPONENT_X, y=19)],
        opponent_policy=OpponentPolicyConfig(type="scripted_advance_and_shoot"),
        terrain=terrain,
        track_exposure=track_exposure,
    )
    return WargameEnv(config, renderer=None)


def _run_episode(env: WargameEnv) -> tuple[float | None, float | None]:
    """Play one episode standing still; return (exposure_rate, proximity)."""
    env.reset(seed=0)
    stay = WargameEnvAction(actions=[0] * N_MODELS)
    terminated = truncated = False
    while not (terminated or truncated):
        _obs, _reward, terminated, truncated, _info = env.step(stay)
    return env.exposure_rate, env.terrain_proximity


def test_open_ground_is_fully_exposed() -> None:
    """Clear line of sight, within weapon range: every model is exposed."""
    env = _make_env()
    try:
        exposure, _proximity = _run_episode(env)
    finally:
        env.close()
    assert exposure == 1.0


def test_terrain_across_the_sightline_removes_exposure() -> None:
    """The whole point: breaking line of sight is what lowers exposure."""
    env = _make_env(terrain=[BLOCKING_RUIN])
    try:
        exposure, proximity = _run_episode(env)
    finally:
        env.close()
    assert exposure == 0.0
    # Players sit at x=10, the ruin starts at x=14.
    assert proximity == pytest.approx(4.0)


def test_out_of_range_is_not_exposed() -> None:
    """Exposure means "could be shot", so range gates it as well as sight."""
    env = _make_env(weapon_range=4)
    try:
        exposure, _proximity = _run_episode(env)
    finally:
        env.close()
    assert exposure == 0.0


def test_untracked_config_reports_no_data_rather_than_zero() -> None:
    """Not measured must not read as never exposed.

    Returning 0.0 here would put an unmeasured run and a perfectly-covered run
    on the same number.
    """
    env = _make_env(track_exposure=False)
    try:
        exposure, proximity = _run_episode(env)
    finally:
        env.close()
    assert exposure is None
    assert proximity is None


def test_proximity_is_undefined_without_terrain() -> None:
    """A board with no terrain has no proximity, and 0.0 would read as cover."""
    env = _make_env()
    try:
        _exposure, proximity = _run_episode(env)
    finally:
        env.close()
    assert proximity is None


def test_distance_to_footprint_is_zero_inside_and_clamped_outside() -> None:
    """Rectangle distance: 0 inside, per-axis overshoot outside."""
    footprints = [Footprint.from_corners(10, 10, 20, 20)]
    positions = np.array([[15, 15], [5, 15], [25, 25], [15, 24]])

    distances = distances_to_nearest_footprint(positions, footprints)

    assert distances[0] == pytest.approx(0.0)  # inside
    assert distances[1] == pytest.approx(5.0)  # left of it
    assert distances[2] == pytest.approx(math.hypot(5, 5))  # diagonal corner
    assert distances[3] == pytest.approx(4.0)  # below it


def test_tracker_ignores_dead_models() -> None:
    """A casualty is not cover: dead models leave the denominator."""
    tracker = ExposureTracker()
    tracker.record(
        exposed=np.array([True, False, True]),
        alive=np.array([True, True, False]),
        terrain_distances=np.array([2.0, 4.0, 99.0]),
    )

    assert tracker.exposure_rate == pytest.approx(0.5)
    assert tracker.terrain_proximity == pytest.approx(3.0)
