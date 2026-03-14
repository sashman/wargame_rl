"""Game timing engine — tracks setup stages, battle rounds, player turns, and phases.

Pure domain logic with no environment or Gym dependencies.
"""

from __future__ import annotations

from wargame_rl.wargame.envs.types.game_timing import (
    BATTLE_PHASE_ORDER,
    SETUP_PHASE_ORDER,
    BattlePhase,
    GamePhase,
    GameState,
    PlayerSide,
)


class GameClockError(RuntimeError):
    """Raised when a clock operation is invalid for the current state."""


def _other_player(side: PlayerSide) -> PlayerSide:
    return PlayerSide.player_2 if side is PlayerSide.player_1 else PlayerSide.player_1


class GameClock:
    """Tracks the current position within the full game timing structure.

    Parameters
    ----------
    n_rounds:
        Number of battle rounds (tabletop standard is 5).
    first_player:
        Which side takes the first player turn each round.
    """

    def __init__(
        self,
        n_rounds: int = 5,
        first_player: PlayerSide = PlayerSide.player_1,
    ) -> None:
        if n_rounds < 1:
            raise ValueError(f"n_rounds must be >= 1, got {n_rounds}")
        self._n_rounds = n_rounds
        self._first_player = first_player
        self._second_player = _other_player(first_player)

        self._game_phase: GamePhase = GamePhase.setup
        self._setup_idx: int = 0
        self._round: int = 1
        self._active_player: PlayerSide = self._first_player
        self._phase_idx: int = 0
        self._total_steps: int = 0

    # -- Properties -----------------------------------------------------------

    @property
    def n_rounds(self) -> int:
        return self._n_rounds

    @property
    def state(self) -> GameState:
        if self._game_phase is GamePhase.setup:
            return GameState(
                game_phase=GamePhase.setup,
                setup_phase=SETUP_PHASE_ORDER[self._setup_idx],
            )
        if self._game_phase is GamePhase.battle:
            return GameState(
                game_phase=GamePhase.battle,
                battle_round=self._round,
                active_player=self._active_player,
                phase=BATTLE_PHASE_ORDER[self._phase_idx],
            )
        return GameState(game_phase=GamePhase.complete)

    @property
    def is_game_over(self) -> bool:
        return self._game_phase is GamePhase.complete

    @property
    def is_setup(self) -> bool:
        return self._game_phase is GamePhase.setup

    @property
    def is_battle(self) -> bool:
        return self._game_phase is GamePhase.battle

    @property
    def total_steps(self) -> int:
        return self._total_steps

    # -- Reset ----------------------------------------------------------------

    def reset(self) -> GameState:
        """Reset the clock to the beginning of setup."""
        self._game_phase = GamePhase.setup
        self._setup_idx = 0
        self._round = 1
        self._active_player = self._first_player
        self._phase_idx = 0
        self._total_steps = 0
        return self.state

    # -- Setup navigation -----------------------------------------------------

    def advance_setup_phase(self) -> GameState:
        """Advance to the next setup stage.

        After the last setup stage, transitions to battle (round 1, first
        player, command phase).
        """
        if self._game_phase is not GamePhase.setup:
            raise GameClockError(
                f"Cannot advance setup phase when game_phase is {self._game_phase.value}"
            )
        self._total_steps += 1
        next_idx = self._setup_idx + 1
        if next_idx >= len(SETUP_PHASE_ORDER):
            self._transition_to_battle()
        else:
            self._setup_idx = next_idx
        return self.state

    def skip_setup(self) -> GameState:
        """Jump from any setup phase directly to the start of battle."""
        if self._game_phase is not GamePhase.setup:
            raise GameClockError(
                f"Cannot skip setup when game_phase is {self._game_phase.value}"
            )
        self._total_steps += 1
        self._transition_to_battle()
        return self.state

    # -- Battle navigation ----------------------------------------------------

    def advance_phase(self) -> GameState:
        """Advance to the next battle phase.

        Rolls over to the next player turn when phases are exhausted, to
        the next round when both player turns are done, and marks the game
        complete after the final phase of the final round.
        """
        self._require_battle("advance_phase")
        self._total_steps += 1
        next_phase_idx = self._phase_idx + 1
        if next_phase_idx < len(BATTLE_PHASE_ORDER):
            self._phase_idx = next_phase_idx
        else:
            self._roll_over_turn()
        return self.state

    def advance_to_phase(self, target: BattlePhase) -> GameState:
        """Skip forward to *target* within the current player turn.

        Raises if *target* is at or before the current phase.
        """
        self._require_battle("advance_to_phase")
        target_idx = BATTLE_PHASE_ORDER.index(target)
        if target_idx <= self._phase_idx:
            current = BATTLE_PHASE_ORDER[self._phase_idx]
            raise GameClockError(
                f"Target phase {target.value} is not ahead of current phase {current.value}"
            )
        steps = target_idx - self._phase_idx
        self._total_steps += steps
        self._phase_idx = target_idx
        return self.state

    def advance_to_next_player_turn(self) -> GameState:
        """Skip remaining phases and move to the next player turn.

        If the current player is the second player, advances to the next
        round (first player, command phase).  Marks complete if no rounds
        remain.
        """
        self._require_battle("advance_to_next_player_turn")
        remaining = len(BATTLE_PHASE_ORDER) - self._phase_idx
        self._total_steps += remaining
        self._roll_over_turn()
        return self.state

    def advance_to_next_round(self) -> GameState:
        """Skip to the start of the next round.

        Skips remaining phases and, if still on the first player's turn,
        also skips the second player's turn.  Marks complete if no rounds
        remain.
        """
        self._require_battle("advance_to_next_round")
        remaining_phases = len(BATTLE_PHASE_ORDER) - self._phase_idx
        if self._active_player is self._first_player:
            remaining_phases += len(BATTLE_PHASE_ORDER)
        self._total_steps += remaining_phases
        self._advance_round()
        return self.state

    # -- Internal helpers -----------------------------------------------------

    def _require_battle(self, method: str) -> None:
        if self._game_phase is GamePhase.complete:
            raise GameClockError(f"Cannot call {method}: game is already over")
        if self._game_phase is GamePhase.setup:
            raise GameClockError(
                f"Cannot call {method} during setup — call advance_setup_phase or skip_setup first"
            )

    def _transition_to_battle(self) -> None:
        self._game_phase = GamePhase.battle
        self._round = 1
        self._active_player = self._first_player
        self._phase_idx = 0

    def _roll_over_turn(self) -> None:
        """Advance from end-of-phases to the next player turn or round."""
        if self._active_player is self._first_player:
            self._active_player = self._second_player
            self._phase_idx = 0
        else:
            self._advance_round()

    def _advance_round(self) -> None:
        next_round = self._round + 1
        if next_round > self._n_rounds:
            self._game_phase = GamePhase.complete
        else:
            self._round = next_round
            self._active_player = self._first_player
            self._phase_idx = 0

    # -- Display --------------------------------------------------------------

    def __repr__(self) -> str:
        if self._game_phase is GamePhase.setup:
            label = SETUP_PHASE_ORDER[self._setup_idx].value.replace("_", " ").title()
            return f"Setup | {label}"
        if self._game_phase is GamePhase.battle:
            phase_label = BATTLE_PHASE_ORDER[self._phase_idx].value.title()
            player_label = self._active_player.value.replace("_", " ").title()
            return f"Round {self._round} | {player_label} | {phase_label}"
        return "Game Complete"
