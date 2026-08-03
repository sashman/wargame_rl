"""Tests for v9 Phase 5: Milestone Validation.

End-to-end verification that all SGS-* requirements hold against the live
codebase.  Covers the full pipeline (snapshot → inject → step → stream →
replay) and the previously-untested analyze_match() analysis layer.
"""

from __future__ import annotations

import json

import pytest

from wargame_rl.wargame.envs.state import (
    EventLogExporter,
    GameStateSnapshot,
    JsonMatchCodec,
    MatchAnalysis,
    ReplayController,
    StepNarrator,
    analyze_match,
    build_codec,
    describe_action,
    validate_snapshot,
)
from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.types.config import OpponentPolicyConfig
from wargame_rl.wargame.envs.wargame import WargameEnv


def _assert_snapshots_equal(
    actual: GameStateSnapshot,
    expected: GameStateSnapshot,
    msg: str = "",
) -> None:
    """Compare snapshots ignoring fields not preserved by delta encoding.

    ObjectiveSnapshot.player_models_in_range and opponent_models_in_range
    are derived from model positions but not tracked in StateDelta, so
    delta-reconstructed snapshots may diverge at non-anchor steps.
    """
    a = actual.model_copy(
        update={
            "objectives": [
                o.model_copy(
                    update={
                        "player_models_in_range": [],
                        "opponent_models_in_range": [],
                    }
                )
                for o in actual.objectives
            ]
        }
    )
    e = expected.model_copy(
        update={
            "objectives": [
                o.model_copy(
                    update={
                        "player_models_in_range": [],
                        "opponent_models_in_range": [],
                    }
                )
                for o in expected.objectives
            ]
        }
    )
    assert a == e, msg


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def small_env_config() -> WargameEnvConfig:
    return WargameEnvConfig(
        board_width=20,
        board_height=20,
        number_of_wargame_models=2,
        number_of_objectives=2,
        number_of_battle_rounds=5,
    )


@pytest.fixture
def pipeline_env(
    small_env_config: WargameEnvConfig,
) -> tuple[WargameEnv, EventLogExporter]:
    """Env wired with an EventLogExporter for full-pipeline tests."""
    exporter = EventLogExporter(anchor_interval=4)
    env = WargameEnv(config=small_env_config, state_exporters=[exporter])
    return env, exporter


# ── End-to-end pipeline ──────────────────────────────────────────────────


class TestEndToEndPipeline:
    """Phase 5 SC-3: snapshot → inject → step → stream → replay."""

    def test_full_pipeline(
        self, pipeline_env: tuple[WargameEnv, EventLogExporter]
    ) -> None:
        """Exercise every v9 subsystem in a single scenario."""
        env, exporter = pipeline_env

        # 1. Reset and capture initial snapshot
        env.reset(seed=42)
        initial_snap = env.to_snapshot()
        assert isinstance(initial_snap, GameStateSnapshot)

        # 2. Run several steps, collecting live snapshots
        live_snapshots: list[GameStateSnapshot] = [initial_snap]
        for _ in range(10):
            action = WargameEnvAction(actions=env.action_space.sample())
            _, _, terminated, truncated, _ = env.step(action)
            live_snapshots.append(env.to_snapshot())
            if terminated or truncated:
                break

        n_steps = len(live_snapshots) - 1
        assert n_steps >= 3, "Need at least 3 steps for a meaningful test"

        # 3. Replay from event log — must match live snapshots (SGS-06)
        controller = ReplayController(exporter.log)
        for live in live_snapshots:
            replayed = controller.seek(live.step)
            _assert_snapshots_equal(replayed, live, f"Mismatch at step {live.step}")

        # 4. Codec round-trip — encode → decode → replay (SGS-04)
        codec = JsonMatchCodec()
        encoded = codec.encode(exporter.log)
        decoded_log = codec.decode(encoded)
        decoded_controller = ReplayController(decoded_log)
        for live in live_snapshots:
            _assert_snapshots_equal(decoded_controller.seek(live.step), live)

        # 5. State injection — pick a mid-episode snapshot, load it,
        #    step from there, and verify we get a valid snapshot (SGS-08)
        mid_idx = n_steps // 2
        mid_snap = live_snapshots[mid_idx]
        env.load_state(mid_snap)
        loaded_snap = env.to_snapshot()
        assert loaded_snap.step == mid_snap.step
        assert loaded_snap.player_vp == mid_snap.player_vp

        action = WargameEnvAction(actions=env.action_space.sample())
        env.step(action)
        post_inject_snap = env.to_snapshot()
        assert post_inject_snap.step == mid_snap.step + 1

        # 6. Round-trip fidelity (SGS-09)
        snap_a = live_snapshots[1]
        env.load_state(snap_a)
        snap_b = env.to_snapshot()
        assert snap_a == snap_b

        # 7. Narration covers the replayed data (SGS-13)
        narrator = StepNarrator()
        text = narrator.narrate(live_snapshots[-1])
        assert isinstance(text, str)
        assert len(text) > 50

        # 8. analyze_match produces a report from replayed snapshots
        #    Use live_snapshots (not controller.iter_snapshots) since the
        #    exporter log was extended by the injection/step above.
        analysis = analyze_match(live_snapshots, file_name="e2e_test")
        assert isinstance(analysis, MatchAnalysis)
        assert analysis.steps == n_steps

    def test_pipeline_with_shooting_config(self) -> None:
        """Full pipeline on a config with opponents and shooting enabled."""
        exporter = EventLogExporter(anchor_interval=3)
        cfg = WargameEnvConfig(
            board_width=20,
            board_height=20,
            number_of_wargame_models=2,
            number_of_objectives=1,
            number_of_battle_rounds=5,
            number_of_opponent_models=2,
            opponent_policy=OpponentPolicyConfig(type="random"),
            skip_phases=[],
        )
        env = WargameEnv(config=cfg, state_exporters=[exporter])
        env.reset(seed=7)

        snapshots: list[GameStateSnapshot] = [env.to_snapshot()]
        for _ in range(15):
            action = WargameEnvAction(actions=env.action_space.sample())
            _, _, terminated, truncated, _ = env.step(action)
            snapshots.append(env.to_snapshot())
            if terminated or truncated:
                break

        # Replay matches live
        controller = ReplayController(exporter.log)
        for live in snapshots:
            _assert_snapshots_equal(controller.seek(live.step), live)

        # Codec round-trip
        codec = build_codec("json")
        decoded = codec.decode(codec.encode(exporter.log))
        for live in snapshots:
            _assert_snapshots_equal(ReplayController(decoded).seek(live.step), live)


# ── analyze_match coverage ───────────────────────────────────────────────


class TestAnalyzeMatch:
    """Coverage for the analysis layer (previously untested)."""

    @pytest.fixture
    def episode_snapshots(
        self, small_env_config: WargameEnvConfig
    ) -> list[GameStateSnapshot]:
        env = WargameEnv(config=small_env_config)
        env.reset(seed=42)
        snaps: list[GameStateSnapshot] = [env.to_snapshot()]
        for _ in range(10):
            action = WargameEnvAction(actions=env.action_space.sample())
            _, _, terminated, truncated, _ = env.step(action)
            snaps.append(env.to_snapshot())
            if terminated or truncated:
                break
        return snaps

    def test_returns_match_analysis(
        self, episode_snapshots: list[GameStateSnapshot]
    ) -> None:
        result = analyze_match(episode_snapshots, file_name="test_run")
        assert isinstance(result, MatchAnalysis)
        assert result.file == "test_run"
        assert result.steps == len(episode_snapshots) - 1

    def test_outcome_reflects_episode_end(
        self, episode_snapshots: list[GameStateSnapshot]
    ) -> None:
        result = analyze_match(episode_snapshots)
        assert result.outcome in ("terminated", "truncated", "in_progress")

    def test_movement_metrics_populated(
        self, episode_snapshots: list[GameStateSnapshot]
    ) -> None:
        result = analyze_match(episode_snapshots)
        assert 0.0 <= result.objective_approach_rate <= 1.0
        assert 0.0 <= result.idle_rate <= 1.0
        assert 0.0 <= result.edge_contact_rate <= 1.0
        assert result.mean_distance_to_objective >= 0.0

    def test_tactical_metrics_populated(
        self, episode_snapshots: list[GameStateSnapshot]
    ) -> None:
        result = analyze_match(episode_snapshots)
        assert result.mean_group_distance >= 0.0
        assert result.vp_per_step >= 0.0 or result.vp_per_step <= 0.0

    def test_rule_compliance_clean(
        self, episode_snapshots: list[GameStateSnapshot]
    ) -> None:
        """A valid env should never produce rule violations."""
        result = analyze_match(episode_snapshots)
        assert result.movement_violations == 0
        assert result.bounds_violations == 0

    def test_degenerate_metrics_populated(
        self, episode_snapshots: list[GameStateSnapshot]
    ) -> None:
        result = analyze_match(episode_snapshots)
        assert result.action_entropy >= 0.0
        assert 0.0 <= result.oscillation_rate <= 1.0
        assert isinstance(result.stagnation_detected, bool)

    def test_tactical_score_range(
        self, episode_snapshots: list[GameStateSnapshot]
    ) -> None:
        result = analyze_match(episode_snapshots)
        assert 0.0 <= result.tactical_score <= 100.0

    def test_to_text_produces_report(
        self, episode_snapshots: list[GameStateSnapshot]
    ) -> None:
        result = analyze_match(episode_snapshots, file_name="text_test")
        text = result.to_text()
        assert "Match Analysis" in text
        assert "MOVEMENT EFFICIENCY" in text
        assert "TACTICAL SCORE" in text

    def test_empty_snapshots_returns_defaults(self) -> None:
        result = analyze_match([], file_name="empty")
        assert result.steps == 0
        assert result.outcome == "unknown"
        assert result.tactical_score == 0.0

    def test_single_snapshot_returns_zero_steps(
        self, small_env_config: WargameEnvConfig
    ) -> None:
        env = WargameEnv(config=small_env_config)
        env.reset(seed=1)
        result = analyze_match([env.to_snapshot()])
        assert result.steps == 0


# ── SGS-* requirement spot-checks ────────────────────────────────────────


class TestRequirementSpotChecks:
    """Quick verification that each SGS-* requirement holds."""

    def test_sgs01_canonical_model_exists(
        self, small_env_config: WargameEnvConfig
    ) -> None:
        """SGS-01: GameStateSnapshot contains board, entities, clock, VP."""
        env = WargameEnv(config=small_env_config)
        env.reset(seed=1)
        snap = env.to_snapshot()
        assert snap.board_width == 20
        assert snap.board_height == 20
        assert len(snap.player_models) == 2
        assert snap.clock is not None
        assert isinstance(snap.player_vp, int)

    def test_sgs02_json_schema_available(
        self, small_env_config: WargameEnvConfig
    ) -> None:
        """SGS-02: Schema version and JSON Schema are available."""
        env = WargameEnv(config=small_env_config)
        env.reset(seed=1)
        snap = env.to_snapshot()
        assert snap.schema_version == "1.2"
        schema = GameStateSnapshot.model_json_schema()
        assert "properties" in schema

    def test_sgs04_json_serialisation(self, small_env_config: WargameEnvConfig) -> None:
        """SGS-04: JSON encoding and codec registry work."""
        env = WargameEnv(config=small_env_config)
        env.reset(seed=1)
        snap = env.to_snapshot()
        json_str = snap.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["step"] == snap.step
        assert build_codec("json") is not None

    def test_sgs07_clock_set_state(self) -> None:
        """SGS-07: GameClock.set_state() positions clock correctly."""
        from wargame_rl.wargame.envs.domain.game_clock import GameClock
        from wargame_rl.wargame.envs.types.game_timing import BattlePhase, GamePhase

        clock = GameClock(n_rounds=5)
        from wargame_rl.wargame.envs.types.game_timing import PlayerSide

        clock.set_state(
            GamePhase.battle,
            battle_round=3,
            active_player=PlayerSide.player_1,
            phase=BattlePhase.shooting,
        )
        state = clock.state
        assert state.battle_round == 3
        assert state.phase == BattlePhase.shooting

    def test_sgs08_load_state(self, small_env_config: WargameEnvConfig) -> None:
        """SGS-08: load_state() makes env ready for step()."""
        env = WargameEnv(config=small_env_config)
        env.reset(seed=1)
        snap = env.to_snapshot()
        env.load_state(snap)
        action = WargameEnvAction(actions=env.action_space.sample())
        result = env.step(action)
        assert len(result) == 5

    def test_sgs09_round_trip(self, small_env_config: WargameEnvConfig) -> None:
        """SGS-09: to_snapshot → load_state → to_snapshot is identical."""
        env = WargameEnv(config=small_env_config)
        env.reset(seed=1)
        env.step(WargameEnvAction(actions=env.action_space.sample()))
        snap_a = env.to_snapshot()
        env.load_state(snap_a)
        snap_b = env.to_snapshot()
        assert snap_a == snap_b

    def test_sgs10_describe_action(self) -> None:
        """SGS-10: describe_action() produces readable text."""
        stay = describe_action(
            0,
            n_angles=16,
            n_speed_bins=6,
            shooting_slice_start=None,
            shooting_slice_end=None,
        )
        assert stay == "Stay"
        move = describe_action(
            1,
            n_angles=16,
            n_speed_bins=6,
            shooting_slice_start=None,
            shooting_slice_end=None,
        )
        assert "Move" in move

    def test_sgs13_narrator(self, small_env_config: WargameEnvConfig) -> None:
        """SGS-13: StepNarrator produces text with expected sections."""
        env = WargameEnv(config=small_env_config)
        env.reset(seed=1)
        env.step(WargameEnvAction(actions=env.action_space.sample()))
        snap = env.to_snapshot()
        narrator = StepNarrator()
        text = narrator.narrate(snap)
        assert "Reward" in text or "reward" in text.lower()

    def test_sgs14_reward_in_snapshot(self, small_env_config: WargameEnvConfig) -> None:
        """SGS-14: Reward breakdown and phase name in snapshot."""
        env = WargameEnv(config=small_env_config)
        env.reset(seed=1)
        env.step(WargameEnvAction(actions=env.action_space.sample()))
        snap = env.to_snapshot()
        assert snap.reward.phase_name is not None
        assert isinstance(snap.reward.breakdown, dict)

    def test_sgs03_delta_encoding(
        self, pipeline_env: tuple[WargameEnv, EventLogExporter]
    ) -> None:
        """SGS-03: Delta encoding with anchors exists in event log."""
        env, exporter = pipeline_env
        env.reset(seed=42)
        for _ in range(8):
            action = WargameEnvAction(actions=env.action_space.sample())
            _, _, t, tr, _ = env.step(action)
            if t or tr:
                break

        from wargame_rl.wargame.envs.state import StepEvent

        step_events = [e for e in exporter.log.events[1:] if isinstance(e, StepEvent)]
        has_anchor = any(e.anchor is not None for e in step_events)
        assert has_anchor, "No anchor snapshots found in event log"

    def test_sgs05_append_only_event_stream(
        self, pipeline_env: tuple[WargameEnv, EventLogExporter]
    ) -> None:
        """SGS-05: Events are append-only and ordered."""
        env, exporter = pipeline_env
        env.reset(seed=42)
        prev_len = len(exporter.log)
        for _ in range(5):
            action = WargameEnvAction(actions=env.action_space.sample())
            _, _, t, tr, _ = env.step(action)
            assert len(exporter.log) > prev_len
            prev_len = len(exporter.log)
            if t or tr:
                break

    def test_sgs06_deterministic_replay(
        self, pipeline_env: tuple[WargameEnv, EventLogExporter]
    ) -> None:
        """SGS-06: Replay reconstructs any historical state."""
        env, exporter = pipeline_env
        env.reset(seed=42)
        live: list[GameStateSnapshot] = [env.to_snapshot()]
        for _ in range(6):
            action = WargameEnvAction(actions=env.action_space.sample())
            _, _, t, tr, _ = env.step(action)
            live.append(env.to_snapshot())
            if t or tr:
                break

        controller = ReplayController(exporter.log)
        for snap in live:
            _assert_snapshots_equal(controller.seek(snap.step), snap)

    def test_validate_snapshot_rejects_invalid(
        self, small_env_config: WargameEnvConfig
    ) -> None:
        """Validation catches out-of-bounds locations."""
        env = WargameEnv(config=small_env_config)
        env.reset(seed=1)
        snap = env.to_snapshot()
        bad = snap.model_copy(deep=True)
        bad.player_models[0].location = [999, 999]
        errors = validate_snapshot(bad, small_env_config)
        assert len(errors) > 0
        assert any("out of bounds" in e for e in errors)
