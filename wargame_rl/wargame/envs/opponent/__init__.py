"""Opponent policy module."""

from wargame_rl.wargame.envs.opponent.policy import OpponentPolicy
from wargame_rl.wargame.envs.opponent.registry import (
    build_opponent_policy,
    register_policy,
)

__all__ = [
    "OpponentPolicy",
    "build_opponent_policy",
    "register_policy",
]
