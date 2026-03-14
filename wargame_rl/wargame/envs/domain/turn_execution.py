"""Turn execution service: advance phases, run opponent turns, skip excluded phases."""

from __future__ import annotations

from collections.abc import Callable

from wargame_rl.wargame.envs.domain.game_clock import GameClock
from wargame_rl.wargame.envs.types.game_timing import BattlePhase, GameState, PlayerSide

OnBeforeAdvance = Callable[[GameClock], None]


def _is_opponent_active(clock_state: GameState, player_side: PlayerSide) -> bool:
    return (
        clock_state.active_player is not None
        and clock_state.active_player != player_side
    )


def _should_skip_phase(
    clock_state: GameState, skip_phases: frozenset[BattlePhase]
) -> bool:
    phase = clock_state.phase
    return phase is not None and phase in skip_phases


def run_until_player_phase(
    clock: GameClock,
    skip_phases: frozenset[BattlePhase],
    player_side: PlayerSide,
    apply_opponent_action: Callable[[], None],
    on_before_advance: OnBeforeAdvance | None = None,
) -> None:
    """Skip excluded phases, then run opponent turn if active, until we're on a player phase.

    Matches original order: skip first (so we may land on opponent's turn), then run
    opponent turn, then skip again. Used after reset and at the end of run_after_player_action.
    """
    # Skip past excluded phases until we hit a non-skip phase or opponent's turn
    while (
        not clock.is_game_over
        and not _is_opponent_active(clock.state, player_side)
        and _should_skip_phase(clock.state, skip_phases)
    ):
        if on_before_advance is not None:
            on_before_advance(clock)
        clock.advance_phase()

    # Execute full opponent turn if it's their turn
    while not clock.is_game_over and _is_opponent_active(clock.state, player_side):
        apply_opponent_action()
        if on_before_advance is not None:
            on_before_advance(clock)
        clock.advance_phase()

    # Skip past excluded phases again (e.g. our command phase) until movement or game over
    while (
        not clock.is_game_over
        and not _is_opponent_active(clock.state, player_side)
        and _should_skip_phase(clock.state, skip_phases)
    ):
        if on_before_advance is not None:
            on_before_advance(clock)
        clock.advance_phase()


def run_after_player_action(
    clock: GameClock,
    skip_phases: frozenset[BattlePhase],
    player_side: PlayerSide,
    apply_opponent_action: Callable[[], None],
    on_before_advance: OnBeforeAdvance | None = None,
) -> None:
    """Advance phase once after player action, then run opponent turn and skip phases as needed.

    Call this from step() after applying the player's action.
    """
    if not clock.is_game_over:
        if on_before_advance is not None:
            on_before_advance(clock)
        clock.advance_phase()

    run_until_player_phase(
        clock, skip_phases, player_side, apply_opponent_action, on_before_advance
    )
