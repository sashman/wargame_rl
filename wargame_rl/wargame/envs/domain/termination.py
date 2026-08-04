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
    success_flag: bool,
    all_eliminated: bool = False,
) -> bool:
    """True when the episode should end: elimination, turn limit, clock complete, or success.

    `success_flag` is the caller's verdict on the active reward phase's success
    criteria, already gated on that phase's `terminate_on_success`. It is passed
    in rather than computed here because the criteria are configurable.

    Episode length is governed by the game clock (max_turns is derived from
    number_of_battle_rounds), so there are no post-game "dead" steps.
    """
    if all_eliminated:
        return True
    return current_turn >= max_turns or clock.is_game_over or success_flag
