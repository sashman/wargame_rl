"""Backward-compat re-export of GameClock from domain."""

from wargame_rl.wargame.envs.domain.game_clock import GameClock, GameClockError

__all__ = ["GameClock", "GameClockError"]
