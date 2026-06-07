"""Tests for PlayerAheadOnVPCriteria."""

from __future__ import annotations

from types import SimpleNamespace

from wargame_rl.wargame.envs.reward.criteria.player_ahead_on_vp import (
    PlayerAheadOnVPCriteria,
)
from wargame_rl.wargame.envs.reward.step_context import StepContext


def test_is_successful_player_ahead() -> None:
    """Player VP greater than opponent VP -> success."""
    criteria = PlayerAheadOnVPCriteria()
    view = SimpleNamespace(player_vp=100, opponent_vp=50)
    ctx: StepContext = SimpleNamespace()  # type: ignore[assignment]
    assert criteria.is_successful(view, ctx) is True


def test_is_successful_player_behind() -> None:
    """Player VP less than opponent VP -> not successful."""
    criteria = PlayerAheadOnVPCriteria()
    view = SimpleNamespace(player_vp=50, opponent_vp=100)
    ctx: StepContext = SimpleNamespace()  # type: ignore[assignment]
    assert criteria.is_successful(view, ctx) is False


def test_is_successful_equal_vps() -> None:
    """Equal VP -> not successful (strictly greater required)."""
    criteria = PlayerAheadOnVPCriteria()
    view = SimpleNamespace(player_vp=75, opponent_vp=75)
    ctx: StepContext = SimpleNamespace()  # type: ignore[assignment]
    assert criteria.is_successful(view, ctx) is False


def test_vp_threshold_for_terminal_bonus_is_none() -> None:
    """This criteria does not gate a terminal VP bonus."""
    criteria = PlayerAheadOnVPCriteria()
    view = SimpleNamespace()
    assert criteria.vp_threshold_for_terminal_bonus(view) is None
