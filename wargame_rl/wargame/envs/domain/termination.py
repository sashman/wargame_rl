"""Termination domain logic: when is the battle over?"""

from __future__ import annotations

from wargame_rl.wargame.envs.domain.game_clock import GameClock


def check_max_turns_reached(current_turn: int, max_turns: int) -> bool:
    """True if the turn limit has been reached."""
    return current_turn >= max_turns


def is_battle_over(
    clock: GameClock,
    current_turn: int,
    max_turns: int,
    max_turns_override: int | None,
    all_models_at_objectives_flag: bool,
    all_eliminated: bool = False,
) -> bool:
    """True when the episode should end: elimination, turn limit, clock complete, or all at objectives.

    When max_turns_override is set (e.g. for training), clock.is_game_over is not
    considered; otherwise game can end by clock or by success.
    """
    if all_eliminated:
        return True
    if max_turns_override is not None:
        return current_turn >= max_turns or all_models_at_objectives_flag
    return (
        current_turn >= max_turns or clock.is_game_over or all_models_at_objectives_flag
    )
