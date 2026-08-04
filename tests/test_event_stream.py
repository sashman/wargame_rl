"""Tests for v9 Phase 4: Event Streaming & Replay (SGS-03, SGS-05, SGS-06)."""

from __future__ import annotations

from pathlib import Path

import pytest

from wargame_rl.wargame.envs.state import (
    EventLog,
    EventLogExporter,
    GameStateSnapshot,
    JsonMatchCodec,
    ReplayController,
    ResetEvent,
    StateExporter,
    StepEvent,
    apply_delta,
    build_codec,
    compute_delta,
)
from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.event_log_callback import EventLogCallback


@pytest.fixture
def env_with_exporter() -> tuple[WargameEnv, EventLogExporter]:
    """Env wired with an EventLogExporter for recording."""
    exporter = EventLogExporter(anchor_interval=5)
    cfg = WargameEnvConfig(
        board_width=20,
        board_height=20,
        number_of_wargame_models=2,
        number_of_objectives=1,
        number_of_battle_rounds=5,
    )
    env = WargameEnv(config=cfg, state_exporters=[exporter])
    return env, exporter


@pytest.fixture
def recorded_log(
    env_with_exporter: tuple[WargameEnv, EventLogExporter],
) -> EventLog:
    """Run a short episode and return the populated EventLog."""
    env, exporter = env_with_exporter
    env.reset(seed=42)
    for _ in range(12):
        action = WargameEnvAction(actions=env.action_space.sample())
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return exporter.log


class TestStateExporterProtocol:
    """Verify StateExporter protocol conformance."""

    def test_event_log_exporter_satisfies_protocol(self) -> None:
        exporter = EventLogExporter()
        assert isinstance(exporter, StateExporter)

    def test_custom_exporter_satisfies_protocol(self) -> None:
        class CustomExporter:
            def on_reset(self, snapshot: GameStateSnapshot) -> None:
                pass

            def on_step(self, snapshot: GameStateSnapshot) -> None:
                pass

        assert isinstance(CustomExporter(), StateExporter)


class TestEventLog:
    """SGS-05: Append-only ordered event stream for a complete match."""

    def test_reset_creates_initial_event(
        self, env_with_exporter: tuple[WargameEnv, EventLogExporter]
    ) -> None:
        env, exporter = env_with_exporter
        env.reset(seed=42)
        log = exporter.log
        assert len(log) == 1
        assert isinstance(log.events[0], ResetEvent)

    def test_steps_append_events(self, recorded_log: EventLog) -> None:
        assert len(recorded_log) > 1
        for event in recorded_log.events[1:]:
            assert isinstance(event, StepEvent)

    def test_events_are_ordered(self, recorded_log: EventLog) -> None:
        steps = []
        for event in recorded_log.events:
            if isinstance(event, ResetEvent):
                steps.append(event.snapshot.step)
            else:
                assert isinstance(event, StepEvent)
                steps.append(event.delta.step)
        assert steps == sorted(steps)

    def test_record_step_before_reset_raises(self) -> None:
        log = EventLog()
        cfg = WargameEnvConfig(
            board_width=10, board_height=10, number_of_wargame_models=1
        )
        env = WargameEnv(config=cfg)
        env.reset(seed=1)
        snapshot = env.to_snapshot()
        with pytest.raises(
            RuntimeError, match="record_step called before record_reset"
        ):
            log.record_step(snapshot)


class TestDeltaEncoding:
    """SGS-03: Layered change protocol with full snapshots and granular deltas."""

    def test_anchors_inserted_at_interval(self, recorded_log: EventLog) -> None:
        anchors = [
            e
            for e in recorded_log.events[1:]
            if isinstance(e, StepEvent) and e.anchor is not None
        ]
        assert len(anchors) > 0

    def test_delta_captures_changes(self) -> None:
        cfg = WargameEnvConfig(
            board_width=20,
            board_height=20,
            number_of_wargame_models=2,
            number_of_objectives=1,
            number_of_battle_rounds=5,
        )
        env = WargameEnv(config=cfg)
        env.reset(seed=42)
        snap_before = env.to_snapshot()
        action = WargameEnvAction(actions=env.action_space.sample())
        env.step(action)
        snap_after = env.to_snapshot()

        delta = compute_delta(snap_before, snap_after)
        assert delta.step == snap_after.step

    def test_apply_delta_reconstructs_state(self) -> None:
        cfg = WargameEnvConfig(
            board_width=20,
            board_height=20,
            number_of_wargame_models=2,
            number_of_objectives=1,
            number_of_battle_rounds=5,
        )
        env = WargameEnv(config=cfg)
        env.reset(seed=42)
        snap_before = env.to_snapshot()
        action = WargameEnvAction(actions=env.action_space.sample())
        env.step(action)
        snap_after = env.to_snapshot()

        delta = compute_delta(snap_before, snap_after)
        reconstructed = apply_delta(snap_before, delta)

        assert reconstructed.step == snap_after.step
        assert reconstructed.clock == snap_after.clock
        assert reconstructed.player_vp == snap_after.player_vp
        assert reconstructed.opponent_vp == snap_after.opponent_vp
        for i, (r, e) in enumerate(
            zip(reconstructed.player_models, snap_after.player_models)
        ):
            assert r.location == e.location, f"player model {i} location mismatch"
            assert r.alive == e.alive, f"player model {i} alive mismatch"

    def test_delta_is_minimal(self) -> None:
        """Delta for identical snapshots has no model deltas."""
        cfg = WargameEnvConfig(
            board_width=20,
            board_height=20,
            number_of_wargame_models=1,
            number_of_objectives=1,
        )
        env = WargameEnv(config=cfg)
        env.reset(seed=42)
        snap = env.to_snapshot()
        delta = compute_delta(snap, snap)
        assert delta.player_model_deltas == []
        assert delta.opponent_model_deltas == []
        assert delta.clock is None
        assert delta.player_vp is None


class TestReplay:
    """SGS-06: Deterministic replay from event log."""

    def test_replay_seek_to_reset_step(self, recorded_log: EventLog) -> None:
        controller = ReplayController(recorded_log)
        first = controller.seek(controller.first_step)
        assert isinstance(first, GameStateSnapshot)
        reset_event = recorded_log.events[0]
        assert isinstance(reset_event, ResetEvent)
        assert first == reset_event.snapshot

    def test_replay_seek_to_last_step(self, recorded_log: EventLog) -> None:
        controller = ReplayController(recorded_log)
        last = controller.seek(controller.last_step)
        assert isinstance(last, GameStateSnapshot)
        assert last.step == controller.last_step

    def test_replay_deterministic_reconstruction(
        self,
        env_with_exporter: tuple[WargameEnv, EventLogExporter],
    ) -> None:
        """Replay must produce identical state to what was recorded."""
        env, exporter = env_with_exporter
        env.reset(seed=99)
        snapshots_direct: list[GameStateSnapshot] = [env.to_snapshot()]
        for _ in range(8):
            action = WargameEnvAction(actions=env.action_space.sample())
            _, _, terminated, truncated, _ = env.step(action)
            snapshots_direct.append(env.to_snapshot())
            if terminated or truncated:
                break

        controller = ReplayController(exporter.log)
        for expected in snapshots_direct:
            reconstructed = controller.seek(expected.step)
            assert reconstructed == expected

    def test_replay_iter_snapshots(self, recorded_log: EventLog) -> None:
        controller = ReplayController(recorded_log)
        all_snaps = controller.iter_snapshots()
        assert len(all_snaps) == len(recorded_log)
        assert all_snaps[0].step == controller.first_step
        assert all_snaps[-1].step == controller.last_step

    def test_iter_snapshots_matches_seek(self, recorded_log: EventLog) -> None:
        """Regression: iter_snapshots() carried reset-time objective occupancy
        for the whole episode because deltas ignored `objectives` and anchors
        were never applied, so it disagreed with seek()."""
        controller = ReplayController(recorded_log)
        for snapshot in controller.iter_snapshots():
            assert snapshot == controller.seek(snapshot.step)

    def test_objective_occupancy_tracked_across_steps(
        self,
        env_with_exporter: tuple[WargameEnv, EventLogExporter],
    ) -> None:
        """Regression: objective occupancy must follow models as they move."""
        env, exporter = env_with_exporter
        env.reset(seed=7)
        direct: list[GameStateSnapshot] = [env.to_snapshot()]
        for _ in range(10):
            action = WargameEnvAction(actions=env.action_space.sample())
            _, _, terminated, truncated, _ = env.step(action)
            direct.append(env.to_snapshot())
            if terminated or truncated:
                break

        replayed = ReplayController(exporter.log).iter_snapshots()
        for expected, actual in zip(direct, replayed):
            assert [o.player_models_in_range for o in actual.objectives] == [
                o.player_models_in_range for o in expected.objectives
            ]
            assert [o.opponent_models_in_range for o in actual.objectives] == [
                o.opponent_models_in_range for o in expected.objectives
            ]

    def test_empty_log_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            ReplayController(EventLog())

    def test_seek_invalid_step_raises(self, recorded_log: EventLog) -> None:
        controller = ReplayController(recorded_log)
        with pytest.raises(ValueError):
            controller.seek(9999)


class TestCodecRoundTrip:
    """Codec serialisation preserves the full event log."""

    def test_json_codec_round_trip(self, recorded_log: EventLog) -> None:
        codec = JsonMatchCodec()
        encoded = codec.encode(recorded_log)
        assert isinstance(encoded, bytes)
        decoded = codec.decode(encoded)

        assert len(decoded) == len(recorded_log)
        controller_orig = ReplayController(recorded_log)
        controller_decoded = ReplayController(decoded)

        for step in range(controller_orig.first_step, controller_orig.last_step + 1):
            orig = controller_orig.seek(step)
            restored = controller_decoded.seek(step)
            assert orig == restored

    def test_json_codec_content_type(self) -> None:
        codec = JsonMatchCodec()
        assert codec.content_type() == "application/x-ndjson"

    def test_build_codec_json(self) -> None:
        codec = build_codec("json")
        assert isinstance(codec, JsonMatchCodec)

    def test_build_codec_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown codec type"):
            build_codec("unknown_format")


class TestEnvIntegration:
    """Env wiring: exporters receive callbacks without affecting the Gym API."""

    def test_env_without_exporters_works(self) -> None:
        cfg = WargameEnvConfig(
            board_width=10,
            board_height=10,
            number_of_wargame_models=1,
            number_of_objectives=1,
        )
        env = WargameEnv(config=cfg)
        obs, info = env.reset(seed=1)
        assert obs is not None
        action = WargameEnvAction(actions=env.action_space.sample())
        result = env.step(action)
        assert len(result) == 5

    def test_multiple_exporters(self) -> None:
        exp1 = EventLogExporter(anchor_interval=3)
        exp2 = EventLogExporter(anchor_interval=7)
        cfg = WargameEnvConfig(
            board_width=10,
            board_height=10,
            number_of_wargame_models=1,
            number_of_objectives=1,
            number_of_battle_rounds=3,
        )
        env = WargameEnv(config=cfg, state_exporters=[exp1, exp2])
        env.reset(seed=1)
        for _ in range(5):
            action = WargameEnvAction(actions=env.action_space.sample())
            env.step(action)

        assert len(exp1.log) == 6
        assert len(exp2.log) == 6


class TestEventLogCallback:
    """The callback must persist a log mid-run, not only after fit() returns."""

    @staticmethod
    def _populated_exporter() -> EventLogExporter:
        exporter = EventLogExporter(anchor_interval=5)
        cfg = WargameEnvConfig(
            board_width=10,
            board_height=10,
            number_of_wargame_models=1,
            number_of_objectives=1,
            number_of_battle_rounds=3,
        )
        env = WargameEnv(config=cfg, state_exporters=[exporter])
        env.reset(seed=1)
        for _ in range(3):
            env.step(WargameEnvAction(actions=env.action_space.sample()))
        return exporter

    def test_epoch_start_writes_a_decodable_log(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A killed run must still leave a readable recording behind."""
        monkeypatch.chdir(tmp_path)
        exporter = self._populated_exporter()
        callback = EventLogCallback("test-run", exporter)

        callback.on_train_epoch_start(None, None)  # type: ignore[arg-type]

        assert callback.output_path.exists()
        decoded = JsonMatchCodec().decode(callback.output_path.read_bytes())
        assert len(decoded) == len(exporter.log)

    def test_write_is_a_noop_before_any_episode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An epoch that recorded nothing must not leave an empty file."""
        monkeypatch.chdir(tmp_path)
        callback = EventLogCallback("test-run", EventLogExporter())

        assert callback.write() is False
        assert not callback.output_path.exists()

    def test_reset_only_log_is_not_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A just-reset episode must not overwrite a usable recording.

        Regression: writing at this moment produced a 1-event file that decoded
        fine but contained no steps to analyse.
        """
        monkeypatch.chdir(tmp_path)
        exporter = EventLogExporter(anchor_interval=5)
        cfg = WargameEnvConfig(
            board_width=10,
            board_height=10,
            number_of_wargame_models=1,
            number_of_objectives=1,
            number_of_battle_rounds=3,
        )
        env = WargameEnv(config=cfg, state_exporters=[exporter])
        env.reset(seed=1)

        callback = EventLogCallback("test-run", exporter)

        assert len(exporter.log) == 1
        assert callback.write() is False
        assert not callback.output_path.exists()
