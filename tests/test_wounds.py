"""Tests for wound tracking, elimination, and termination."""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.envs.domain.entities import WargameModel, alive_mask_for
from wargame_rl.wargame.envs.domain.game_clock import GameClock
from wargame_rl.wargame.envs.domain.termination import is_battle_over
from wargame_rl.wargame.envs.types.config import ModelConfig


def _make_model(max_wounds: int, current_wounds: int | None = None) -> WargameModel:
    """Helper: create a WargameModel with wound stats."""
    cw = current_wounds if current_wounds is not None else max_wounds
    return WargameModel(
        location=np.array([0, 0], dtype=np.int32),
        stats={"max_wounds": max_wounds, "current_wounds": cw},
        distances_to_objectives=np.zeros((1, 2), dtype=np.int32),
        group_id=0,
    )


@pytest.mark.parametrize(
    "max_wounds, damage, expected_current",
    [
        (1, 1, 0),
        (2, 1, 1),
        (3, 2, 1),
    ],
)
def test_wound_tracking(
    max_wounds: int, damage: int, expected_current: int
) -> None:
    model = _make_model(max_wounds)
    model.take_damage(damage)
    assert model.stats["current_wounds"] == expected_current


def test_take_damage_clamps_at_zero() -> None:
    model = _make_model(max_wounds=2)
    model.take_damage(10)
    assert model.stats["current_wounds"] == 0


def test_is_alive_true_when_wounded() -> None:
    model = _make_model(max_wounds=2)
    assert model.is_alive is True
    model.take_damage(1)
    assert model.is_alive is True


def test_is_alive_false_when_eliminated() -> None:
    model = _make_model(max_wounds=1)
    model.take_damage(1)
    assert model.is_alive is False


def test_reset_for_episode_restores_wounds() -> None:
    model = _make_model(max_wounds=3)
    model.take_damage(2)
    assert model.stats["current_wounds"] == 1
    model.reset_for_episode()
    assert model.stats["current_wounds"] == 3
    assert model.is_alive is True


def test_is_battle_over_all_eliminated() -> None:
    clock = GameClock(n_rounds=5)
    result = is_battle_over(
        clock=clock,
        current_turn=0,
        max_turns=100,
        max_turns_override=100,
        all_models_at_objectives_flag=False,
        all_eliminated=True,
    )
    assert result is True


def test_is_battle_over_without_elimination() -> None:
    clock = GameClock(n_rounds=5)
    result = is_battle_over(
        clock=clock,
        current_turn=0,
        max_turns=100,
        max_turns_override=100,
        all_models_at_objectives_flag=False,
        all_eliminated=False,
    )
    assert result is False


def test_config_default_max_wounds() -> None:
    assert ModelConfig().max_wounds == 1


def test_config_explicit_max_wounds() -> None:
    assert ModelConfig(max_wounds=3).max_wounds == 3


def test_alive_mask_for() -> None:
    alive = _make_model(max_wounds=2)
    dead = _make_model(max_wounds=1)
    dead.take_damage(1)
    result = alive_mask_for([alive, dead])
    np.testing.assert_array_equal(result, np.array([True, False]))
