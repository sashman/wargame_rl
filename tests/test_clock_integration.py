"""Tests for GameClock integration with WargameEnv."""

from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.types.game_timing import BattlePhase, GamePhase
from wargame_rl.wargame.envs.wargame import WargameEnv


def _make_env(n_rounds: int = 10, **overrides: object) -> WargameEnv:
    defaults: dict = dict(
        render_mode=None,
        board_width=20,
        board_height=20,
        number_of_wargame_models=2,
        number_of_objectives=1,
        objective_radius_size=2,
        number_of_battle_rounds=n_rounds,
    )
    defaults.update(overrides)
    return WargameEnv(config=WargameEnvConfig(**defaults))


class TestClockLifecycleOnReset:
    def test_reset_initialises_clock_to_movement_phase(self) -> None:
        env = _make_env()
        env.reset(seed=42)
        state = env._game_clock.state
        assert state.game_phase is GamePhase.battle
        assert state.phase is BattlePhase.movement
        assert state.battle_round == 1

    def test_reset_clears_clock_after_episode(self) -> None:
        env = _make_env(n_rounds=3)
        env.reset(seed=42)
        for _ in range(3):
            env.step(WargameEnvAction(actions=[0, 0]))
        assert env._game_clock.is_game_over

        env.reset(seed=42)
        assert not env._game_clock.is_game_over
        assert env._game_clock.state.battle_round == 1


class TestClockAdvancesPerStep:
    def test_step_advances_round(self) -> None:
        env = _make_env(n_rounds=10)
        env.reset(seed=42)
        assert env._game_clock.state.battle_round == 1

        env.step(WargameEnvAction(actions=[0, 0]))
        assert env._game_clock.state.battle_round == 2

        env.step(WargameEnvAction(actions=[0, 0]))
        assert env._game_clock.state.battle_round == 3

    def test_step_stays_on_movement_phase(self) -> None:
        env = _make_env(n_rounds=5)
        env.reset(seed=42)
        for _ in range(4):
            env.step(WargameEnvAction(actions=[0, 0]))
            assert env._game_clock.state.phase is BattlePhase.movement


class TestTermination:
    def test_terminates_after_n_rounds(self) -> None:
        env = _make_env(n_rounds=3)
        env.reset(seed=42)
        for i in range(3):
            _, _, terminated, _, _ = env.step(WargameEnvAction(actions=[0, 0]))
            if i < 2:
                assert not terminated
        assert terminated

    def test_max_turns_equals_n_rounds(self) -> None:
        env = _make_env(n_rounds=7)
        assert env.max_turns == 7

    def test_stepping_after_game_over_stays_terminated(self) -> None:
        env = _make_env(n_rounds=2)
        env.reset(seed=42)
        env.step(WargameEnvAction(actions=[0, 0]))
        _, _, terminated, _, _ = env.step(WargameEnvAction(actions=[0, 0]))
        assert terminated

        _, _, terminated2, _, _ = env.step(WargameEnvAction(actions=[0, 0]))
        assert terminated2


class TestObservationClockFields:
    def test_observation_contains_round_and_phase(self) -> None:
        env = _make_env(n_rounds=10)
        obs, _ = env.reset(seed=42)
        assert obs.battle_round == 1
        assert obs.battle_phase_index == list(BattlePhase).index(BattlePhase.movement)
        assert obs.n_rounds == 10

    def test_observation_round_advances(self) -> None:
        env = _make_env(n_rounds=10)
        env.reset(seed=42)
        obs, _, _, _, _ = env.step(WargameEnvAction(actions=[0, 0]))
        assert obs.battle_round == 2

    def test_game_tensor_size_is_3(self) -> None:
        env = _make_env()
        obs, _ = env.reset(seed=42)
        assert obs.size_game_observation == 3
