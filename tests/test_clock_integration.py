"""Tests for GameClock integration with WargameEnv.

Each env.step() advances one battle phase for the player.  After the
player's five phases (command → fight), the opponent's full turn is
auto-executed before the observation is returned.
"""

from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.types.game_timing import (
    BATTLE_PHASE_ORDER,
    NON_MOVEMENT_PHASES,
    BattlePhase,
    GamePhase,
)
from wargame_rl.wargame.envs.wargame import WargameEnv

N_PHASES = len(BATTLE_PHASE_ORDER)


def _make_env(n_rounds: int = 10, **overrides: object) -> WargameEnv:
    defaults: dict = dict(
        render_mode=None,
        board_width=20,
        board_height=20,
        number_of_wargame_models=2,
        number_of_objectives=1,
        objective_radius_size=2,
        number_of_battle_rounds=n_rounds,
        skip_phases=[],  # step through every phase unless overridden
        max_turns_override=None,  # use game-clock-derived max_turns for clock tests
    )
    defaults.update(overrides)
    return WargameEnv(config=WargameEnvConfig(**defaults))


def _stay(n_models: int = 2) -> WargameEnvAction:
    return WargameEnvAction(actions=[0] * n_models)


class TestClockLifecycleOnReset:
    def test_reset_initialises_clock_to_command_phase(self) -> None:
        env = _make_env()
        env.reset(seed=42)
        state = env._game_clock.state
        assert state.game_phase is GamePhase.battle
        assert state.phase is BattlePhase.command
        assert state.battle_round == 1

    def test_reset_clears_clock_after_episode(self) -> None:
        env = _make_env(n_rounds=3)
        env.reset(seed=42)
        for _ in range(3 * N_PHASES):
            env.step(_stay())
        assert env._game_clock.is_game_over

        env.reset(seed=42)
        assert not env._game_clock.is_game_over
        assert env._game_clock.state.battle_round == 1


class TestClockAdvancesPerStep:
    def test_step_advances_phase(self) -> None:
        env = _make_env(n_rounds=10)
        env.reset(seed=42)
        assert env._game_clock.state.phase is BattlePhase.command

        env.step(_stay())
        assert env._game_clock.state.phase is BattlePhase.movement

        env.step(_stay())
        assert env._game_clock.state.phase is BattlePhase.shooting

    def test_five_steps_advances_round(self) -> None:
        """After 5 steps (one full player turn), the round advances."""
        env = _make_env(n_rounds=10)
        env.reset(seed=42)
        assert env._game_clock.state.battle_round == 1

        for _ in range(N_PHASES):
            env.step(_stay())

        assert env._game_clock.state.battle_round == 2
        assert env._game_clock.state.phase is BattlePhase.command

    def test_phase_cycles_through_all(self) -> None:
        env = _make_env(n_rounds=10)
        env.reset(seed=42)
        for i, expected_phase in enumerate(BATTLE_PHASE_ORDER):
            assert env._game_clock.state.phase is expected_phase, (
                f"Step {i}: expected {expected_phase}, got {env._game_clock.state.phase}"
            )
            env.step(_stay())


class TestTermination:
    def test_terminates_after_n_rounds(self) -> None:
        n_rounds = 3
        env = _make_env(n_rounds=n_rounds)
        env.reset(seed=42)
        total_steps = n_rounds * N_PHASES
        for i in range(total_steps):
            _, _, terminated, _, _ = env.step(_stay())
            if i < total_steps - 1:
                assert not terminated
        assert terminated

    def test_max_turns_equals_n_rounds_times_phases(self) -> None:
        env = _make_env(n_rounds=7)
        assert env.max_turns == 7 * N_PHASES

    def test_stepping_after_game_over_stays_terminated(self) -> None:
        env = _make_env(n_rounds=2)
        env.reset(seed=42)
        for _ in range(2 * N_PHASES):
            env.step(_stay())
        assert env._game_clock.is_game_over

        _, _, terminated, _, _ = env.step(_stay())
        assert terminated


class TestSkipPhases:
    """Tests for skip_phases (default skips all non-movement phases)."""

    def test_default_skips_non_movement(self) -> None:
        """Config default skips command/shooting/charge/fight."""
        cfg = WargameEnvConfig(
            render_mode=None,
            board_width=20,
            board_height=20,
            number_of_wargame_models=2,
            number_of_objectives=1,
            objective_radius_size=2,
            max_turns_override=None,
        )
        env = WargameEnv(config=cfg)
        env.reset(seed=42)
        assert env._game_clock.state.phase is BattlePhase.movement
        assert env.max_turns == cfg.number_of_battle_rounds

    def test_reset_lands_on_movement(self) -> None:
        env = _make_env(n_rounds=10, skip_phases=list(NON_MOVEMENT_PHASES))
        env.reset(seed=42)
        assert env._game_clock.state.phase is BattlePhase.movement

    def test_step_returns_to_movement(self) -> None:
        env = _make_env(n_rounds=10, skip_phases=list(NON_MOVEMENT_PHASES))
        env.reset(seed=42)
        env.step(_stay())
        assert env._game_clock.state.phase is BattlePhase.movement

    def test_one_step_per_round(self) -> None:
        env = _make_env(n_rounds=10, skip_phases=list(NON_MOVEMENT_PHASES))
        env.reset(seed=42)
        assert env._game_clock.state.battle_round == 1
        env.step(_stay())
        assert env._game_clock.state.battle_round == 2

    def test_max_turns_equals_n_rounds(self) -> None:
        env = _make_env(n_rounds=7, skip_phases=list(NON_MOVEMENT_PHASES))
        assert env.max_turns == 7

    def test_terminates_after_n_rounds(self) -> None:
        n_rounds = 3
        env = _make_env(n_rounds=n_rounds, skip_phases=list(NON_MOVEMENT_PHASES))
        env.reset(seed=42)
        for i in range(n_rounds):
            _, _, terminated, _, _ = env.step(_stay())
            if i < n_rounds - 1:
                assert not terminated
        assert terminated

    def test_observation_phase_is_movement(self) -> None:
        env = _make_env(n_rounds=10, skip_phases=list(NON_MOVEMENT_PHASES))
        obs, _ = env.reset(seed=42)
        assert obs.battle_phase_index == list(BattlePhase).index(BattlePhase.movement)
        obs, _, _, _, _ = env.step(_stay())
        assert obs.battle_phase_index == list(BattlePhase).index(BattlePhase.movement)

    def test_empty_skip_list_steps_all_phases(self) -> None:
        env = _make_env(n_rounds=10, skip_phases=[])
        env.reset(seed=42)
        assert env._game_clock.state.phase is BattlePhase.command
        assert env.max_turns == 10 * N_PHASES

    def test_partial_skip(self) -> None:
        """Skipping only command keeps 4 phases per round."""
        env = _make_env(n_rounds=5, skip_phases=[BattlePhase.command])
        env.reset(seed=42)
        assert env._game_clock.state.phase is BattlePhase.movement
        assert env.max_turns == 5 * 4


class TestObservationClockFields:
    def test_observation_contains_round_and_phase(self) -> None:
        env = _make_env(n_rounds=10)
        obs, _ = env.reset(seed=42)
        assert obs.battle_round == 1
        assert obs.battle_phase_index == list(BattlePhase).index(BattlePhase.command)
        assert obs.n_rounds == 10

    def test_observation_round_advances_after_full_turn(self) -> None:
        env = _make_env(n_rounds=10)
        env.reset(seed=42)
        for _ in range(N_PHASES):
            obs, _, _, _, _ = env.step(_stay())
        assert obs.battle_round == 2

    def test_observation_phase_advances_each_step(self) -> None:
        env = _make_env(n_rounds=10)
        env.reset(seed=42)
        obs, _, _, _, _ = env.step(_stay())
        assert obs.battle_phase_index == list(BattlePhase).index(BattlePhase.movement)

    def test_game_tensor_size_includes_vp(self) -> None:
        env = _make_env()
        obs, _ = env.reset(seed=42)
        # 5: placeholder + round + phase + player_vp + opponent_vp
        assert obs.size_game_observation == 5
