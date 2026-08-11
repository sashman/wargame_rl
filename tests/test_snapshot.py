"""Integration tests for the canonical GameStateSnapshot."""

from __future__ import annotations

import json

import pytest

from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.envs.state.snapshot import GameStateSnapshot, JsonEncoder
from wargame_rl.wargame.envs.types import TurnOrder, WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.types.config import (
    ModelConfig,
    OpponentPolicyConfig,
    TerrainPieceConfig,
    WeaponProfile,
)
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv


@pytest.fixture
def shooting_env() -> WargameEnv:
    """Env with opponents, weapons, and only movement+shooting phases."""
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
    env = WargameEnv(config=cfg)
    return env


def _step_to_shooting(env: WargameEnv) -> None:
    """Step with STAY until we reach shooting phase."""
    n = len(env.wargame_models)
    env.step(WargameEnvAction(actions=[STAY_ACTION] * n))


class TestSnapshotAfterReset:
    def test_to_snapshot_after_reset(self, shooting_env: WargameEnv) -> None:
        shooting_env.reset(seed=42)
        snap = shooting_env.to_snapshot()

        assert snap.board_width == 30
        assert snap.board_height == 30
        assert len(snap.player_models) == 2
        assert len(snap.opponent_models) == 2
        assert len(snap.objectives) == 1
        assert snap.step == 0
        assert snap.player_actions is None
        assert snap.player_action_descriptions is None
        assert snap.is_terminated is False


class TestSnapshotAfterStep:
    def test_to_snapshot_after_step(self, shooting_env: WargameEnv) -> None:
        shooting_env.reset(seed=42)
        n = len(shooting_env.wargame_models)
        shooting_env.step(WargameEnvAction(actions=[STAY_ACTION] * n))
        snap = shooting_env.to_snapshot()

        assert snap.step >= 1
        assert snap.player_actions is not None

    def test_action_phase_is_the_phase_that_acted(
        self, shooting_env: WargameEnv
    ) -> None:
        """Regression: `clock.battle_phase` is the *next* phase because the clock
        advances before the snapshot is taken, so actions could not be attributed
        to a phase. `action_phase` records the phase that actually executed."""
        shooting_env.reset(seed=42)
        n = len(shooting_env.wargame_models)

        shooting_env.step(WargameEnvAction(actions=[STAY_ACTION] * n))
        first = shooting_env.to_snapshot()
        assert first.action_phase == BattlePhase.movement.value
        assert first.clock.battle_phase != first.action_phase

        shooting_env.step(WargameEnvAction(actions=[STAY_ACTION] * n))
        second = shooting_env.to_snapshot()
        assert second.action_phase == BattlePhase.shooting.value

    def test_action_phase_is_none_after_reset(self, shooting_env: WargameEnv) -> None:
        shooting_env.reset(seed=42)
        assert shooting_env.to_snapshot().action_phase is None

    def test_opponent_actions_recorded(self, shooting_env: WargameEnv) -> None:
        """Opponent acts after player finishes all phases in a round."""
        shooting_env.reset(seed=42)
        n = len(shooting_env.wargame_models)
        # Step through movement then shooting to complete player's turn
        shooting_env.step(WargameEnvAction(actions=[STAY_ACTION] * n))
        shooting_env.step(WargameEnvAction(actions=[STAY_ACTION] * n))
        snap = shooting_env.to_snapshot()

        assert snap.opponent_actions is not None


class TestJsonSerialisation:
    def test_json_serialisation(self, shooting_env: WargameEnv) -> None:
        shooting_env.reset(seed=42)
        snap = shooting_env.to_snapshot()
        raw = snap.model_dump_json()
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)
        assert "player_models" in parsed

    def test_json_schema(self) -> None:
        schema = GameStateSnapshot.model_json_schema()
        assert "properties" in schema


class TestCombatResults:
    def test_combat_results_have_pairing(self, shooting_env: WargameEnv) -> None:
        """After a shooting step, combat results carry attacker/target indices and analytical fields."""
        shooting_env.reset(seed=42)
        _step_to_shooting(shooting_env)

        ss = shooting_env._action_handler.shooting_slice
        assert ss is not None
        shooting_env.step(WargameEnvAction(actions=[ss.start, ss.start + 1]))

        snap = shooting_env.to_snapshot()
        if snap.player_combat_results:
            cr = snap.player_combat_results[0]
            assert cr.attacker_idx >= 0
            assert cr.target_idx >= 0
            assert cr.expected_damage >= 0.0
            assert 0.0 <= cr.hit_probability <= 1.0
            assert 0.0 <= cr.wound_probability <= 1.0


class TestRewardBreakdown:
    def test_reward_breakdown_in_snapshot(self, shooting_env: WargameEnv) -> None:
        shooting_env.reset(seed=42)
        n = len(shooting_env.wargame_models)
        shooting_env.step(WargameEnvAction(actions=[STAY_ACTION] * n))
        snap = shooting_env.to_snapshot()

        assert snap.reward.phase_name != ""
        assert snap.reward.phase_index >= 0


class TestObjectiveControl:
    def test_objective_control(self, shooting_env: WargameEnv) -> None:
        shooting_env.reset(seed=42)
        snap = shooting_env.to_snapshot()

        assert isinstance(snap.objective_control, list)
        assert len(snap.objective_control) == len(snap.objectives)
        for ctrl in snap.objective_control:
            assert ctrl in ("player", "opponent", "none")


class TestSchemaVersion:
    def test_schema_version(self, shooting_env: WargameEnv) -> None:
        shooting_env.reset(seed=42)
        snap = shooting_env.to_snapshot()
        assert snap.schema_version == "2.3"


class TestEncoder:
    def test_encoder(self, shooting_env: WargameEnv) -> None:
        shooting_env.reset(seed=42)
        snap = shooting_env.to_snapshot()
        encoder = JsonEncoder()
        raw = encoder.encode(snap)
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)
        assert encoder.content_type() == "application/json"


class TestSpatialFields:
    def test_model_distances_to_objectives(self, shooting_env: WargameEnv) -> None:
        shooting_env.reset(seed=42)
        snap = shooting_env.to_snapshot()

        for pm in snap.player_models:
            assert len(pm.distances_to_objectives) == len(snap.objectives)
            assert all(d >= 0.0 for d in pm.distances_to_objectives)

        for om in snap.opponent_models:
            assert len(om.distances_to_objectives) == len(snap.objectives)

    def test_at_objective_flags(self, shooting_env: WargameEnv) -> None:
        shooting_env.reset(seed=42)
        snap = shooting_env.to_snapshot()

        for pm in snap.player_models:
            assert len(pm.at_objective) == len(snap.objectives)
            assert all(isinstance(v, bool) for v in pm.at_objective)

    def test_closest_objective(self, shooting_env: WargameEnv) -> None:
        shooting_env.reset(seed=42)
        snap = shooting_env.to_snapshot()

        for pm in snap.player_models:
            if pm.alive:
                assert pm.closest_objective_idx is not None
                assert pm.closest_objective_distance is not None
                assert 0 <= pm.closest_objective_idx < len(snap.objectives)
            else:
                assert pm.closest_objective_idx is None

    def test_objective_models_in_range(self, shooting_env: WargameEnv) -> None:
        shooting_env.reset(seed=42)
        snap = shooting_env.to_snapshot()

        for obj in snap.objectives:
            assert isinstance(obj.player_models_in_range, list)
            assert isinstance(obj.opponent_models_in_range, list)
            for idx in obj.player_models_in_range:
                assert 0 <= idx < len(snap.player_models)
            for idx in obj.opponent_models_in_range:
                assert 0 <= idx < len(snap.opponent_models)


class TestForceBalance:
    def test_alive_counts(self, shooting_env: WargameEnv) -> None:
        shooting_env.reset(seed=42)
        snap = shooting_env.to_snapshot()

        assert snap.player_alive_count == sum(1 for m in snap.player_models if m.alive)
        assert snap.opponent_alive_count == sum(
            1 for m in snap.opponent_models if m.alive
        )

    def test_total_wounds(self, shooting_env: WargameEnv) -> None:
        shooting_env.reset(seed=42)
        snap = shooting_env.to_snapshot()

        assert snap.player_total_wounds == sum(
            m.current_wounds for m in snap.player_models if m.alive
        )
        assert snap.opponent_total_wounds == sum(
            m.current_wounds for m in snap.opponent_models if m.alive
        )


class TestActionDescriptions:
    def test_action_descriptions_after_step(self, shooting_env: WargameEnv) -> None:
        shooting_env.reset(seed=42)
        n = len(shooting_env.wargame_models)
        shooting_env.step(WargameEnvAction(actions=[STAY_ACTION] * n))
        snap = shooting_env.to_snapshot()

        assert snap.player_action_descriptions is not None
        assert len(snap.player_action_descriptions) == n
        assert all(d == "Stay" for d in snap.player_action_descriptions)

    def test_shooting_action_description(self, shooting_env: WargameEnv) -> None:
        shooting_env.reset(seed=42)
        _step_to_shooting(shooting_env)

        ss = shooting_env._action_handler.shooting_slice
        assert ss is not None
        shooting_env.step(WargameEnvAction(actions=[ss.start, ss.start + 1]))
        snap = shooting_env.to_snapshot()

        assert snap.player_action_descriptions is not None
        assert snap.player_action_descriptions[0] == "Shoot at opponent 0"
        assert snap.player_action_descriptions[1] == "Shoot at opponent 1"


class TestMissionContext:
    def test_mission_fields(self, shooting_env: WargameEnv) -> None:
        shooting_env.reset(seed=42)
        snap = shooting_env.to_snapshot()

        assert isinstance(snap.mission_type, str)
        assert isinstance(snap.mission_params, dict)


class TestTerrainFootprints:
    """Schema 2.1: static terrain geometry rides on the full snapshot."""

    @staticmethod
    def _terrain_env() -> WargameEnv:
        cfg = WargameEnvConfig(
            board_width=20,
            board_height=20,
            number_of_wargame_models=2,
            number_of_objectives=1,
            terrain=[
                TerrainPieceConfig(footprint=(5, 5, 8, 8)),
                TerrainPieceConfig(footprint=(12, 12, 15, 16)),
            ],
        )
        return WargameEnv(config=cfg)

    def test_footprints_match_env_terrain(self) -> None:
        env = self._terrain_env()
        env.reset(seed=1)
        snap = env.to_snapshot()

        assert snap.schema_version == "2.3"
        assert snap.terrain_footprints is not None
        assert len(snap.terrain_footprints) == len(env.terrain.footprints)
        for recorded, footprint in zip(snap.terrain_footprints, env.terrain.footprints):
            assert recorded == footprint.polygon.vertices.tolist()

    def test_footprints_survive_json(self) -> None:
        env = self._terrain_env()
        env.reset(seed=1)
        snap = env.to_snapshot()

        restored = GameStateSnapshot.model_validate(json.loads(snap.model_dump_json()))
        assert restored.terrain_footprints == snap.terrain_footprints

    def test_none_without_terrain(self) -> None:
        cfg = WargameEnvConfig(
            board_width=20,
            board_height=20,
            number_of_wargame_models=2,
            number_of_objectives=1,
        )
        env = WargameEnv(config=cfg)
        env.reset(seed=1)

        assert env.to_snapshot().terrain_footprints is None
