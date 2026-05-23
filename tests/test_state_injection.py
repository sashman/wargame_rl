"""Tests for state injection: GameClock.set_state, validate_snapshot, load_state, round-trip."""

from __future__ import annotations

import pytest

from wargame_rl.wargame.envs.domain.game_clock import GameClock, GameClockError
from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.envs.state.snapshot import validate_snapshot
from wargame_rl.wargame.envs.types import TurnOrder, WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.types.config import (
    ModelConfig,
    OpponentPolicyConfig,
    WeaponProfile,
)
from wargame_rl.wargame.envs.types.game_timing import (
    BattlePhase,
    GamePhase,
    PlayerSide,
    SetupPhase,
)
from wargame_rl.wargame.envs.wargame import WargameEnv


@pytest.fixture
def env() -> WargameEnv:
    """Env with opponents, weapons, and movement+shooting phases."""
    cfg = WargameEnvConfig(
        board_width=30,
        board_height=30,
        number_of_wargame_models=2,
        number_of_objectives=1,
        number_of_opponent_models=2,
        models=[
            ModelConfig(
                x=5,
                y=5,
                max_wounds=3,
                weapons=[
                    WeaponProfile(
                        range=50,
                        attacks=4,
                        ballistic_skill=2,
                        strength=8,
                        ap=2,
                        damage=2,
                    )
                ],
            ),
            ModelConfig(
                x=6,
                y=5,
                max_wounds=3,
                weapons=[
                    WeaponProfile(
                        range=50,
                        attacks=4,
                        ballistic_skill=2,
                        strength=8,
                        ap=2,
                        damage=2,
                    )
                ],
            ),
        ],
        opponent_models=[
            ModelConfig(
                x=20,
                y=5,
                max_wounds=3,
                weapons=[WeaponProfile(range=50)],
            ),
            ModelConfig(
                x=21,
                y=5,
                max_wounds=3,
                weapons=[WeaponProfile(range=50)],
            ),
        ],
        opponent_policy=OpponentPolicyConfig(type="random"),
        turn_order=TurnOrder.player,
        skip_phases=[BattlePhase.command, BattlePhase.charge, BattlePhase.fight],
        n_movement_angles=8,
        n_speed_bins=3,
    )
    return WargameEnv(config=cfg)


# ---------------------------------------------------------------------------
# GameClock.set_state
# ---------------------------------------------------------------------------


class TestClockSetState:
    def test_set_to_battle_round_2_shooting(self) -> None:
        clock = GameClock(n_rounds=5)
        state = clock.set_state(
            GamePhase.battle,
            battle_round=2,
            active_player=PlayerSide.player_1,
            phase=BattlePhase.shooting,
            total_steps=10,
        )
        assert state.game_phase is GamePhase.battle
        assert state.battle_round == 2
        assert state.active_player is PlayerSide.player_1
        assert state.phase is BattlePhase.shooting
        assert clock.total_steps == 10

    def test_set_to_setup(self) -> None:
        clock = GameClock(n_rounds=5)
        clock.skip_setup()
        state = clock.set_state(
            GamePhase.setup,
            setup_phase=SetupPhase.deploy_armies,
        )
        assert state.game_phase is GamePhase.setup
        assert state.setup_phase is SetupPhase.deploy_armies

    def test_set_to_complete(self) -> None:
        clock = GameClock(n_rounds=5)
        state = clock.set_state(GamePhase.complete)
        assert state.game_phase is GamePhase.complete
        assert clock.is_game_over

    def test_battle_missing_round_raises(self) -> None:
        clock = GameClock(n_rounds=5)
        with pytest.raises(GameClockError, match="battle_round.*required"):
            clock.set_state(
                GamePhase.battle,
                active_player=PlayerSide.player_1,
                phase=BattlePhase.movement,
            )

    def test_battle_round_out_of_range_raises(self) -> None:
        clock = GameClock(n_rounds=5)
        with pytest.raises(GameClockError, match="battle_round must be in"):
            clock.set_state(
                GamePhase.battle,
                battle_round=6,
                active_player=PlayerSide.player_1,
                phase=BattlePhase.movement,
            )

    def test_setup_missing_phase_raises(self) -> None:
        clock = GameClock(n_rounds=5)
        with pytest.raises(GameClockError, match="setup_phase is required"):
            clock.set_state(GamePhase.setup)

    def test_advance_works_after_set_state(self) -> None:
        clock = GameClock(n_rounds=5)
        clock.set_state(
            GamePhase.battle,
            battle_round=3,
            active_player=PlayerSide.player_1,
            phase=BattlePhase.command,
            total_steps=20,
        )
        state = clock.advance_phase()
        assert state.phase is BattlePhase.movement
        assert clock.total_steps == 21


# ---------------------------------------------------------------------------
# validate_snapshot
# ---------------------------------------------------------------------------


class TestValidateSnapshot:
    def test_valid_snapshot_passes(self, env: WargameEnv) -> None:
        env.reset(seed=42)
        snap = env.to_snapshot()
        errors = validate_snapshot(snap, env.config)
        assert errors == []

    def test_wrong_player_model_count(self, env: WargameEnv) -> None:
        env.reset(seed=42)
        snap = env.to_snapshot()
        bad = snap.model_copy(update={"player_models": snap.player_models[:1]})
        errors = validate_snapshot(bad, env.config)
        assert any("player models" in e for e in errors)

    def test_wrong_opponent_model_count(self, env: WargameEnv) -> None:
        env.reset(seed=42)
        snap = env.to_snapshot()
        bad = snap.model_copy(update={"opponent_models": []})
        errors = validate_snapshot(bad, env.config)
        assert any("opponent models" in e for e in errors)

    def test_out_of_bounds_location(self, env: WargameEnv) -> None:
        env.reset(seed=42)
        snap = env.to_snapshot()
        bad_models = [m.model_copy() for m in snap.player_models]
        bad_models[0] = bad_models[0].model_copy(update={"location": [99, 99]})
        bad = snap.model_copy(update={"player_models": bad_models})
        errors = validate_snapshot(bad, env.config)
        assert any("out of bounds" in e for e in errors)

    def test_wounds_out_of_range(self, env: WargameEnv) -> None:
        env.reset(seed=42)
        snap = env.to_snapshot()
        bad_models = [m.model_copy() for m in snap.player_models]
        bad_models[0] = bad_models[0].model_copy(update={"current_wounds": 999})
        bad = snap.model_copy(update={"player_models": bad_models})
        errors = validate_snapshot(bad, env.config)
        assert any("current_wounds" in e for e in errors)

    def test_invalid_step(self, env: WargameEnv) -> None:
        env.reset(seed=42)
        snap = env.to_snapshot()
        bad = snap.model_copy(update={"step": -1})
        errors = validate_snapshot(bad, env.config)
        assert any("step=" in e for e in errors)


# ---------------------------------------------------------------------------
# WargameEnv.load_state
# ---------------------------------------------------------------------------


class TestLoadState:
    def test_load_after_reset(self, env: WargameEnv) -> None:
        env.reset(seed=42)
        snap = env.to_snapshot()
        obs, info = env.load_state(snap)
        assert obs is not None
        assert isinstance(info, dict)

    def test_models_at_expected_positions(self, env: WargameEnv) -> None:
        env.reset(seed=42)
        snap = env.to_snapshot()
        env.reset(seed=99)
        env.load_state(snap)

        for i, ms in enumerate(snap.player_models):
            assert list(env.wargame_models[i].location) == ms.location
        for i, ms in enumerate(snap.opponent_models):
            assert list(env.opponent_models[i].location) == ms.location

    def test_wounds_restored(self, env: WargameEnv) -> None:
        env.reset(seed=42)
        snap = env.to_snapshot()
        bad_models = [m.model_copy() for m in snap.player_models]
        bad_models[0] = bad_models[0].model_copy(update={"current_wounds": 1})
        modified_snap = snap.model_copy(update={"player_models": bad_models})
        env.load_state(modified_snap)
        assert env.wargame_models[0].stats["current_wounds"] == 1

    def test_vp_restored(self, env: WargameEnv) -> None:
        env.reset(seed=42)
        snap = env.to_snapshot()
        modified_snap = snap.model_copy(update={"player_vp": 10, "opponent_vp": 5})
        env.load_state(modified_snap)
        assert env.player_vp == 10
        assert env.opponent_vp == 5

    def test_invalid_snapshot_raises(self, env: WargameEnv) -> None:
        env.reset(seed=42)
        snap = env.to_snapshot()
        bad = snap.model_copy(update={"player_models": []})
        with pytest.raises(ValueError, match="Invalid snapshot"):
            env.load_state(bad)


# ---------------------------------------------------------------------------
# Round-trip fidelity
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_round_trip_after_reset(self, env: WargameEnv) -> None:
        env.reset(seed=42)
        snap1 = env.to_snapshot()
        env.load_state(snap1)
        snap2 = env.to_snapshot()

        assert snap1.model_dump() == snap2.model_dump()

    def test_round_trip_after_step(self, env: WargameEnv) -> None:
        env.reset(seed=42)
        n = len(env.wargame_models)
        env.step(WargameEnvAction(actions=[STAY_ACTION] * n))
        snap1 = env.to_snapshot()

        env.reset(seed=99)
        env.load_state(snap1)
        snap2 = env.to_snapshot()

        assert snap1.model_dump() == snap2.model_dump()

    def test_round_trip_with_shooting(self, env: WargameEnv) -> None:
        env.reset(seed=42)
        n = len(env.wargame_models)
        env.step(WargameEnvAction(actions=[STAY_ACTION] * n))

        ss = env._action_handler.shooting_slice
        assert ss is not None
        env.step(WargameEnvAction(actions=[ss.start, ss.start + 1]))
        snap1 = env.to_snapshot()

        env.reset(seed=99)
        env.load_state(snap1)
        snap2 = env.to_snapshot()

        assert snap1.model_dump() == snap2.model_dump()


# ---------------------------------------------------------------------------
# Load state then step
# ---------------------------------------------------------------------------


class TestLoadStateThenStep:
    def test_step_after_load(self, env: WargameEnv) -> None:
        """Env is steppable immediately after load_state."""
        env.reset(seed=42)
        snap = env.to_snapshot()

        env.reset(seed=99)
        env.load_state(snap)

        n = len(env.wargame_models)
        obs, reward, terminated, truncated, info = env.step(
            WargameEnvAction(actions=[STAY_ACTION] * n)
        )
        assert obs is not None
        assert isinstance(reward, float)

    def test_step_produces_valid_snapshot(self, env: WargameEnv) -> None:
        """A snapshot taken after load+step is also valid."""
        env.reset(seed=42)
        snap = env.to_snapshot()

        env.reset(seed=99)
        env.load_state(snap)

        n = len(env.wargame_models)
        env.step(WargameEnvAction(actions=[STAY_ACTION] * n))
        snap2 = env.to_snapshot()

        errors = validate_snapshot(snap2, env.config)
        assert errors == []
        assert snap2.step == snap.step + 1
