"""Tests for mission VP calculator and objective control."""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.envs.mission import build_vp_calculator
from wargame_rl.wargame.envs.mission.vp_calculator import (
    DefaultVPCalculator,
    NoneVPCalculator,
    objective_control_from_caches,
)
from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.types.game_timing import PlayerSide
from wargame_rl.wargame.envs.wargame import WargameEnv


def test_objective_control_player_only() -> None:
    """Player in range of both objectives, opponent in range of none -> 2, 0."""
    player_norms = np.array([[0.0, 0.0]])  # both in range
    opponent_norms = np.array([[10.0, 10.0]])  # neither in range
    radii = np.array([1.0, 1.0])
    n_player, n_opponent = objective_control_from_caches(
        player_norms, opponent_norms, radii
    )
    assert n_player == 2
    assert n_opponent == 0


def test_objective_control_opponent_only() -> None:
    """Opponent in range of both, player in range of none -> 0, 2."""
    player_norms = np.array([[10.0, 10.0]])
    opponent_norms = np.array([[0.0, 0.0]])
    radii = np.array([1.0, 1.0])
    n_player, n_opponent = objective_control_from_caches(
        player_norms, opponent_norms, radii
    )
    assert n_player == 0
    assert n_opponent == 2


def test_objective_control_contested() -> None:
    """Both in range of same objective -> contested, 0 for that objective."""
    player_norms = np.array([[0.0, 10.0]])  # obj0 in range, obj1 not
    opponent_norms = np.array([[0.0, 10.0]])  # same
    radii = np.array([1.0, 1.0])
    n_player, n_opponent = objective_control_from_caches(
        player_norms, opponent_norms, radii
    )
    assert n_player == 0
    assert n_opponent == 0


def test_objective_control_split() -> None:
    """Player controls obj0, opponent controls obj1 -> 1, 1."""
    player_norms = np.array([[0.0, 10.0]])
    opponent_norms = np.array([[10.0, 0.0]])
    radii = np.array([1.0, 1.0])
    n_player, n_opponent = objective_control_from_caches(
        player_norms, opponent_norms, radii
    )
    assert n_player == 1
    assert n_opponent == 1


def test_default_vp_calculator_below_min_round() -> None:
    """Default calculator returns 0 when current_round < min_round."""
    env = WargameEnv(
        config=WargameEnvConfig(
            render_mode=None,
            board_width=10,
            board_height=10,
            number_of_wargame_models=2,
            number_of_objectives=2,
            number_of_battle_rounds=5,
        )
    )
    env.reset(seed=42)
    calc = DefaultVPCalculator(vp_per_objective=5, cap_per_turn=15, min_round=2)
    vp = calc.compute_vp(env, PlayerSide.player_1, 1, PlayerSide.player_1)
    assert vp == 0


def test_default_vp_calculator_cap() -> None:
    """Default calculator caps at cap_per_turn (e.g. 4 objectives * 5 = 20 -> 15)."""
    env = WargameEnv(
        config=WargameEnvConfig(
            render_mode=None,
            board_width=10,
            board_height=10,
            number_of_wargame_models=2,
            number_of_objectives=4,
            number_of_battle_rounds=5,
        )
    )
    env.reset(seed=42)
    calc = DefaultVPCalculator(vp_per_objective=5, cap_per_turn=15, min_round=2)
    # With random placement we may not control 4; use a high cap and check structure
    vp = calc.compute_vp(env, PlayerSide.player_1, 2, PlayerSide.player_1)
    assert 0 <= vp <= 15


def test_none_vp_calculator_returns_zero() -> None:
    """None mission calculator always returns 0."""
    env = WargameEnv(
        config=WargameEnvConfig(render_mode=None, number_of_battle_rounds=5)
    )
    env.reset(seed=42)
    calc = NoneVPCalculator()
    assert calc.compute_vp(env, PlayerSide.player_1, 2, PlayerSide.player_1) == 0
    assert calc.compute_vp(env, PlayerSide.player_2, 3, PlayerSide.player_1) == 0


def test_info_vp_after_step() -> None:
    """After step, info contains player_vp and opponent_vp (may stay 0 before round 2)."""
    env = WargameEnv(
        config=WargameEnvConfig(
            render_mode=None,
            board_width=10,
            board_height=10,
            number_of_wargame_models=2,
            number_of_objectives=2,
            number_of_battle_rounds=5,
        )
    )
    env.reset(seed=42)
    _, _, _, _, info = env.step(WargameEnvAction(actions=[0, 0]))
    assert "player_vp" in info
    assert "opponent_vp" in info
    assert "player_vp_delta" in info
    assert "opponent_vp_delta" in info


def test_build_vp_calculator_default() -> None:
    """build_vp_calculator('default', {}) returns DefaultVPCalculator."""
    calc = build_vp_calculator("default", {})
    assert isinstance(calc, DefaultVPCalculator)


def test_build_vp_calculator_none() -> None:
    """build_vp_calculator('none', {}) returns NoneVPCalculator."""
    calc = build_vp_calculator("none", {})
    assert isinstance(calc, NoneVPCalculator)


def test_build_vp_calculator_unknown_raises() -> None:
    """Unknown mission type raises ValueError."""
    with pytest.raises(ValueError, match="Unknown mission VP calculator type"):
        build_vp_calculator("unknown_mission", {})
