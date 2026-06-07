"""Tests for LastNTurnsCriteria."""

from unittest.mock import Mock

from wargame_rl.wargame.envs.reward.criteria.last_turn import LastTurnCriteria


def test_is_successful_player_wins() -> None:
    """Test that criteria returns True when player VP is greater than opponent VP."""
    criteria = LastTurnCriteria()

    # Mock view where player wins
    mock_view = Mock()
    mock_view.player_vp = 100
    mock_view.opponent_vp = 50

    result = criteria.is_successful(mock_view, Mock())
    assert result is True


def test_is_successful_player_loses() -> None:
    """Test that criteria returns False when player VP is less than or equal to opponent VP."""
    criteria = LastTurnCriteria()

    # Mock view where player loses
    mock_view = Mock()
    mock_view.player_vp = 50
    mock_view.opponent_vp = 100

    result = criteria.is_successful(mock_view, Mock())
    assert result is False


def test_is_successful_equal_vps() -> None:
    """Test that criteria returns False when player VP equals opponent VP."""
    criteria = LastTurnCriteria()

    # Mock view where VPs are equal
    mock_view = Mock()
    mock_view.player_vp = 75
    mock_view.opponent_vp = 75

    result = criteria.is_successful(mock_view, Mock())
    assert result is False


def test_vp_threshold_for_terminal_bonus() -> None:
    """Test that vp_threshold_for_terminal_bonus returns None."""
    criteria = LastTurnCriteria()

    mock_view = Mock()
    result = criteria.vp_threshold_for_terminal_bonus(mock_view)
    assert result is None
