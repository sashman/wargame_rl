"""Comprehensive tests for the GameClock timing engine."""

from __future__ import annotations

import pytest

from wargame_rl.wargame.envs.env_components.game_clock import GameClock, GameClockError
from wargame_rl.wargame.envs.types.game_timing import (
    BATTLE_PHASE_ORDER,
    SETUP_PHASE_ORDER,
    BattlePhase,
    GamePhase,
    PlayerSide,
    SetupPhase,
)


@pytest.fixture
def clock() -> GameClock:
    return GameClock(n_rounds=5)


# ---------------------------------------------------------------------------
# Setup phase tests
# ---------------------------------------------------------------------------


class TestSetupTraversal:
    def test_initial_state_is_setup(self, clock: GameClock) -> None:
        assert clock.is_setup
        assert not clock.is_battle
        assert not clock.is_game_over
        state = clock.state
        assert state.game_phase is GamePhase.setup
        assert state.setup_phase is SetupPhase.muster_armies

    def test_advance_through_all_setup_phases(self, clock: GameClock) -> None:
        visited: list[SetupPhase] = [clock.state.setup_phase]  # type: ignore[list-item]
        for _ in range(len(SETUP_PHASE_ORDER) - 1):
            state = clock.advance_setup_phase()
            if state.game_phase is GamePhase.setup:
                visited.append(state.setup_phase)  # type: ignore[arg-type]
        assert visited == list(SETUP_PHASE_ORDER)

    def test_last_setup_advance_transitions_to_battle(self, clock: GameClock) -> None:
        for _ in range(len(SETUP_PHASE_ORDER)):
            state = clock.advance_setup_phase()
        assert state.game_phase is GamePhase.battle
        assert state.battle_round == 1
        assert state.active_player is PlayerSide.player_1
        assert state.phase is BattlePhase.command

    def test_setup_phase_ordering_matches_tuple(self, clock: GameClock) -> None:
        for expected in SETUP_PHASE_ORDER:
            assert clock.state.setup_phase is expected
            clock.advance_setup_phase()

    def test_cannot_advance_setup_during_battle(self, clock: GameClock) -> None:
        clock.skip_setup()
        with pytest.raises(GameClockError, match="(?i)cannot advance setup"):
            clock.advance_setup_phase()

    def test_cannot_advance_phase_during_setup(self, clock: GameClock) -> None:
        with pytest.raises(GameClockError, match="during setup"):
            clock.advance_phase()


class TestSkipSetup:
    def test_skip_from_first_setup_phase(self, clock: GameClock) -> None:
        state = clock.skip_setup()
        assert state.game_phase is GamePhase.battle
        assert state.battle_round == 1
        assert state.active_player is PlayerSide.player_1
        assert state.phase is BattlePhase.command

    def test_skip_from_middle_setup_phase(self, clock: GameClock) -> None:
        for _ in range(4):
            clock.advance_setup_phase()
        assert clock.state.setup_phase is SetupPhase.declare_battle_formations
        state = clock.skip_setup()
        assert state.game_phase is GamePhase.battle
        assert state.battle_round == 1

    def test_skip_from_last_setup_phase(self, clock: GameClock) -> None:
        for _ in range(len(SETUP_PHASE_ORDER) - 1):
            clock.advance_setup_phase()
        assert clock.state.setup_phase is SetupPhase.resolve_pre_battle_rules
        state = clock.skip_setup()
        assert state.game_phase is GamePhase.battle

    def test_cannot_skip_setup_during_battle(self, clock: GameClock) -> None:
        clock.skip_setup()
        with pytest.raises(GameClockError, match="(?i)cannot skip setup"):
            clock.skip_setup()


# ---------------------------------------------------------------------------
# Battle phase tests
# ---------------------------------------------------------------------------


class TestBattlePhaseOrdering:
    def test_phases_in_correct_order_within_turn(self, clock: GameClock) -> None:
        clock.skip_setup()
        for expected_phase in BATTLE_PHASE_ORDER:
            assert clock.state.phase is expected_phase
            clock.advance_phase()

    def test_five_battle_phases_per_player_turn(self, clock: GameClock) -> None:
        clock.skip_setup()
        phases: list[BattlePhase] = []
        for _ in range(len(BATTLE_PHASE_ORDER)):
            phases.append(clock.state.phase)  # type: ignore[arg-type]
            clock.advance_phase()
        assert phases == list(BATTLE_PHASE_ORDER)


class TestPlayerTurnAlternation:
    def test_player_1_then_player_2(self, clock: GameClock) -> None:
        clock.skip_setup()
        assert clock.state.active_player is PlayerSide.player_1
        for _ in range(len(BATTLE_PHASE_ORDER)):
            clock.advance_phase()
        assert clock.state.active_player is PlayerSide.player_2
        assert clock.state.phase is BattlePhase.command

    def test_full_round_alternation(self, clock: GameClock) -> None:
        clock.skip_setup()
        phases_per_turn = len(BATTLE_PHASE_ORDER)
        for _ in range(phases_per_turn):
            assert clock.state.active_player is PlayerSide.player_1
            clock.advance_phase()
        for _ in range(phases_per_turn):
            assert clock.state.active_player is PlayerSide.player_2
            clock.advance_phase()
        assert clock.state.battle_round == 2
        assert clock.state.active_player is PlayerSide.player_1


class TestRoundRollover:
    def test_round_increments_after_both_turns(self, clock: GameClock) -> None:
        clock.skip_setup()
        phases_per_round = 2 * len(BATTLE_PHASE_ORDER)
        for _ in range(phases_per_round):
            clock.advance_phase()
        assert clock.state.battle_round == 2
        assert clock.state.active_player is PlayerSide.player_1
        assert clock.state.phase is BattlePhase.command


class TestFullGameTraversal:
    def test_50_phases_for_5_rounds(self, clock: GameClock) -> None:
        clock.skip_setup()
        total_phases = 5 * 2 * len(BATTLE_PHASE_ORDER)
        for i in range(total_phases - 1):
            assert not clock.is_game_over, f"Game ended early at phase {i}"
            clock.advance_phase()
        clock.advance_phase()
        assert clock.is_game_over
        assert clock.state.game_phase is GamePhase.complete

    def test_state_is_none_fields_when_complete(self, clock: GameClock) -> None:
        clock.skip_setup()
        total = 5 * 2 * len(BATTLE_PHASE_ORDER)
        for _ in range(total):
            clock.advance_phase()
        state = clock.state
        assert state.setup_phase is None
        assert state.battle_round is None
        assert state.active_player is None
        assert state.phase is None


# ---------------------------------------------------------------------------
# Skip / jump methods
# ---------------------------------------------------------------------------


class TestAdvanceToPhase:
    def test_skip_forward(self, clock: GameClock) -> None:
        clock.skip_setup()
        assert clock.state.phase is BattlePhase.command
        state = clock.advance_to_phase(BattlePhase.shooting)
        assert state.phase is BattlePhase.shooting
        assert state.active_player is PlayerSide.player_1
        assert state.battle_round == 1

    def test_skip_to_last_phase(self, clock: GameClock) -> None:
        clock.skip_setup()
        state = clock.advance_to_phase(BattlePhase.fight)
        assert state.phase is BattlePhase.fight

    def test_error_if_target_is_current(self, clock: GameClock) -> None:
        clock.skip_setup()
        with pytest.raises(GameClockError, match="not ahead"):
            clock.advance_to_phase(BattlePhase.command)

    def test_error_if_target_is_behind(self, clock: GameClock) -> None:
        clock.skip_setup()
        clock.advance_to_phase(BattlePhase.shooting)
        with pytest.raises(GameClockError, match="not ahead"):
            clock.advance_to_phase(BattlePhase.movement)

    def test_total_steps_accounts_for_skipped(self, clock: GameClock) -> None:
        clock.skip_setup()
        clock.advance_to_phase(BattlePhase.fight)
        assert clock.total_steps == 1 + 4  # skip_setup + 4 phases skipped


class TestAdvanceToNextPlayerTurn:
    def test_from_first_phase(self, clock: GameClock) -> None:
        clock.skip_setup()
        state = clock.advance_to_next_player_turn()
        assert state.active_player is PlayerSide.player_2
        assert state.phase is BattlePhase.command
        assert state.battle_round == 1

    def test_from_middle_phase(self, clock: GameClock) -> None:
        clock.skip_setup()
        clock.advance_to_phase(BattlePhase.shooting)
        state = clock.advance_to_next_player_turn()
        assert state.active_player is PlayerSide.player_2
        assert state.phase is BattlePhase.command

    def test_from_second_player_goes_to_next_round(self, clock: GameClock) -> None:
        clock.skip_setup()
        clock.advance_to_next_player_turn()  # -> P2
        state = clock.advance_to_next_player_turn()  # -> next round P1
        assert state.active_player is PlayerSide.player_1
        assert state.battle_round == 2
        assert state.phase is BattlePhase.command

    def test_last_player_last_round_completes(self, clock: GameClock) -> None:
        clock.skip_setup()
        for _ in range(9):
            clock.advance_to_next_player_turn()
        state = clock.advance_to_next_player_turn()
        assert state.game_phase is GamePhase.complete


class TestAdvanceToNextRound:
    def test_from_first_player(self, clock: GameClock) -> None:
        clock.skip_setup()
        state = clock.advance_to_next_round()
        assert state.battle_round == 2
        assert state.active_player is PlayerSide.player_1
        assert state.phase is BattlePhase.command

    def test_from_second_player(self, clock: GameClock) -> None:
        clock.skip_setup()
        clock.advance_to_next_player_turn()  # -> P2
        state = clock.advance_to_next_round()
        assert state.battle_round == 2
        assert state.active_player is PlayerSide.player_1

    def test_last_round_completes(self, clock: GameClock) -> None:
        clock.skip_setup()
        for _ in range(4):
            clock.advance_to_next_round()
        state = clock.advance_to_next_round()
        assert state.game_phase is GamePhase.complete


# ---------------------------------------------------------------------------
# General tests
# ---------------------------------------------------------------------------


class TestGameOverGuard:
    def test_advance_phase_raises(self, clock: GameClock) -> None:
        clock.skip_setup()
        for _ in range(5):
            clock.advance_to_next_round()
        assert clock.is_game_over
        with pytest.raises(GameClockError, match="game is already over"):
            clock.advance_phase()

    def test_advance_to_phase_raises(self, clock: GameClock) -> None:
        clock.skip_setup()
        for _ in range(5):
            clock.advance_to_next_round()
        with pytest.raises(GameClockError, match="game is already over"):
            clock.advance_to_phase(BattlePhase.movement)

    def test_advance_to_next_player_turn_raises(self, clock: GameClock) -> None:
        clock.skip_setup()
        for _ in range(5):
            clock.advance_to_next_round()
        with pytest.raises(GameClockError, match="game is already over"):
            clock.advance_to_next_player_turn()

    def test_advance_to_next_round_raises(self, clock: GameClock) -> None:
        clock.skip_setup()
        for _ in range(5):
            clock.advance_to_next_round()
        with pytest.raises(GameClockError, match="game is already over"):
            clock.advance_to_next_round()


class TestReset:
    def test_reset_returns_to_setup(self, clock: GameClock) -> None:
        clock.skip_setup()
        for _ in range(10):
            clock.advance_phase()
        state = clock.reset()
        assert state.game_phase is GamePhase.setup
        assert state.setup_phase is SetupPhase.muster_armies
        assert clock.total_steps == 0
        assert clock.is_setup
        assert not clock.is_battle
        assert not clock.is_game_over

    def test_reset_after_game_over(self, clock: GameClock) -> None:
        clock.skip_setup()
        for _ in range(5):
            clock.advance_to_next_round()
        assert clock.is_game_over
        state = clock.reset()
        assert state.game_phase is GamePhase.setup
        assert not clock.is_game_over


class TestTotalSteps:
    def test_counts_advance_phase_calls(self, clock: GameClock) -> None:
        clock.skip_setup()
        for i in range(7):
            clock.advance_phase()
            assert clock.total_steps == 1 + (i + 1)  # 1 for skip_setup

    def test_counts_setup_advances(self, clock: GameClock) -> None:
        for i in range(3):
            clock.advance_setup_phase()
        assert clock.total_steps == 3

    def test_combined_setup_and_battle(self, clock: GameClock) -> None:
        for _ in range(len(SETUP_PHASE_ORDER)):
            clock.advance_setup_phase()
        setup_steps = len(SETUP_PHASE_ORDER)
        clock.advance_phase()
        assert clock.total_steps == setup_steps + 1


class TestCustomNRounds:
    def test_one_round(self) -> None:
        clock = GameClock(n_rounds=1)
        clock.skip_setup()
        total = 2 * len(BATTLE_PHASE_ORDER)
        for _ in range(total):
            clock.advance_phase()
        assert clock.is_game_over

    def test_three_rounds(self) -> None:
        clock = GameClock(n_rounds=3)
        clock.skip_setup()
        total = 3 * 2 * len(BATTLE_PHASE_ORDER)
        for _ in range(total):
            clock.advance_phase()
        assert clock.is_game_over

    def test_invalid_n_rounds(self) -> None:
        with pytest.raises(ValueError, match="n_rounds must be >= 1"):
            GameClock(n_rounds=0)


class TestCustomFirstPlayer:
    def test_player_2_goes_first(self) -> None:
        clock = GameClock(first_player=PlayerSide.player_2)
        clock.skip_setup()
        assert clock.state.active_player is PlayerSide.player_2
        for _ in range(len(BATTLE_PHASE_ORDER)):
            clock.advance_phase()
        assert clock.state.active_player is PlayerSide.player_1

    def test_player_2_first_round_rollover(self) -> None:
        clock = GameClock(n_rounds=2, first_player=PlayerSide.player_2)
        clock.skip_setup()
        clock.advance_to_next_round()
        assert clock.state.battle_round == 2
        assert clock.state.active_player is PlayerSide.player_2


class TestFullGameIncludingSetup:
    def test_end_to_end(self) -> None:
        clock = GameClock(n_rounds=2)
        assert clock.is_setup
        for _ in range(len(SETUP_PHASE_ORDER)):
            clock.advance_setup_phase()
        assert clock.is_battle
        total_battle = 2 * 2 * len(BATTLE_PHASE_ORDER)
        for _ in range(total_battle):
            clock.advance_phase()
        assert clock.is_game_over
        expected_steps = len(SETUP_PHASE_ORDER) + total_battle
        assert clock.total_steps == expected_steps


class TestRepr:
    def test_setup_repr(self, clock: GameClock) -> None:
        assert repr(clock) == "Setup | Muster Armies"

    def test_battle_repr(self, clock: GameClock) -> None:
        clock.skip_setup()
        clock.advance_phase()
        assert repr(clock) == "Round 1 | Player 1 | Movement"

    def test_complete_repr(self, clock: GameClock) -> None:
        clock.skip_setup()
        for _ in range(5):
            clock.advance_to_next_round()
        assert repr(clock) == "Game Complete"
