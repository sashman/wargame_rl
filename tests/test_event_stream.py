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
from wargame_rl.wargame.envs.state.events import _POSITION_EPSILON
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


def assert_snapshots_agree(
    actual: GameStateSnapshot, expected: GameStateSnapshot
) -> None:
    """Assert two snapshots match, allowing positions the encoder's tolerance.

    Everything except model positions is compared exactly. Positions get
    `_POSITION_EPSILON`, which is the delta encoder's own definition of
    "unchanged" -- well below any distance the rules can distinguish, since a
    base is ~1.26 across.
    """
    assert actual.step == expected.step
    for field in ("player_models", "opponent_models"):
        actual_models = getattr(actual, field)
        expected_models = getattr(expected, field)
        assert len(actual_models) == len(expected_models)
        for got, want in zip(actual_models, expected_models):
            for position_field in ("location", "previous_location"):
                a = getattr(got, position_field)
                b = getattr(want, position_field)
                if a is None or b is None:
                    assert a is b
                    continue
                assert all(abs(x - y) <= _POSITION_EPSILON for x, y in zip(a, b)), (
                    f"{field}.{position_field}: {a} vs {b}"
                )
            assert got.model_dump(
                exclude={"location", "previous_location"}
            ) == want.model_dump(exclude={"location", "previous_location"})
    assert actual.model_dump(
        exclude={"player_models", "opponent_models"}
    ) == expected.model_dump(exclude={"player_models", "opponent_models"})


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
        """Replay must reproduce recorded state, to the encoder's own tolerance.

        Two things this test used to get wrong, which together made it fail
        about one run in twenty:

        **The action space has its own RNG.** `env.reset(seed=...)` does not
        seed it, so every run drew a different action sequence and the test was
        not deterministic at all. It is seeded here, and seeded to **6**
        specifically because that sequence walks a model into the board-edge
        clamp -- the case below.

        **The delta encoder is lossy on purpose.** `_POSITION_EPSILON` (1e-9)
        exists because under continuous coordinates a stationary model drifts by
        float noise through the clamp and the distance cache, so exact equality
        would make every model emit a delta every step and the compression the
        event log exists for would collapse. Asserting *bit-exact*
        reconstruction therefore asserted something the format never promised:
        a model clamped to the board edge lands 1 ULP from where it started,
        the delta is legitimately dropped, and replay keeps the older value.
        The contract is agreement to within that epsilon, and that is what is
        checked -- exactly, on every other field.
        """
        env, exporter = env_with_exporter
        env.reset(seed=99)
        env.action_space.seed(6)
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
            assert isinstance(reconstructed, GameStateSnapshot)
            assert_snapshots_agree(reconstructed, expected)

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


# ---------------------------------------------------------------------------
# The delta codec covers every dynamic field, by name
# ---------------------------------------------------------------------------

# Static per episode: recorded on the reset and anchor snapshots, never in a
# delta, and preserved by `apply_delta` through `model_copy`.
STATIC_SNAPSHOT_FIELDS = frozenset(
    {
        "schema_version",
        "max_steps",
        "n_rounds",
        "board_width",
        "board_height",
        "deployment_zone",
        "opponent_deployment_zone",
        "terrain_footprints",
        "skip_phases",
        "deployment_outline",
        "opponent_deployment_outline",
        "rules",
        "mission_type",
        "mission_params",
    }
)
STATIC_MODEL_FIELDS = frozenset(
    {"base_radius", "group_id", "max_wounds", "toughness", "save", "weapons"}
)


def test_every_dynamic_snapshot_field_has_a_delta_field() -> None:
    """Schema 2.7 added the melee lists to the snapshot and not to the codec,
    so a replay rebuilt from deltas carried the anchor's melee lists on every
    step. This pins the coverage structurally so the next field cannot."""
    from wargame_rl.wargame.envs.state.events import ModelDelta, StateDelta
    from wargame_rl.wargame.envs.state.snapshot import ModelSnapshot

    dynamic = set(GameStateSnapshot.model_fields) - STATIC_SNAPSHOT_FIELDS
    covered = set(StateDelta.model_fields) - {
        "player_model_deltas",
        "opponent_model_deltas",
    } | {"player_models", "opponent_models"}
    assert dynamic == covered, (
        f"snapshot fields without a delta field: {sorted(dynamic - covered)}; "
        f"delta fields without a snapshot field: {sorted(covered - dynamic)}"
    )

    dynamic_model = set(ModelSnapshot.model_fields) - STATIC_MODEL_FIELDS
    covered_model = set(ModelDelta.model_fields) - {"idx"}
    assert dynamic_model == covered_model, (
        f"model fields without a delta field: {sorted(dynamic_model - covered_model)}"
    )


def test_every_dynamic_field_round_trips_through_the_delta(
    recorded_log: EventLog,
) -> None:
    """One mutation per dynamic field, applied and recovered. The table's keys
    are asserted equal to the dynamic set, so a new field fails here by name
    until it gets a mutation."""
    from wargame_rl.wargame.envs.state.snapshot import (
        ClockSnapshot,
        CombatResultSnapshot,
        DecisionSnapshot,
        ModelSnapshot,
        RewardSnapshot,
    )

    base = ReplayController(recorded_log).iter_snapshots()[0]
    result = CombatResultSnapshot(
        attacker_idx=0,
        target_idx=1,
        hits=1,
        wounds=1,
        unsaved=1,
        damage_dealt=1,
        expected_damage=0.5,
        hit_probability=0.5,
        wound_probability=0.5,
        killed=True,
    )
    model = base.player_models[0]
    moved = model.model_copy(update={"location": [model.location[0] + 1.0, 0.0]})
    mutations: dict[str, object] = {
        "step": base.step + 1,
        "clock": ClockSnapshot(
            game_phase="battle",
            battle_round=3,
            active_player="p2",
            battle_phase="fight",
        ),
        "action_phase": "fight",
        "player_models": [moved, *base.player_models[1:]],
        "opponent_models": [
            base.opponent_models[0].model_copy(update={"alive": False}),
            *base.opponent_models[1:],
        ]
        if base.opponent_models
        else [],
        "objectives": [
            o.model_copy(update={"player_models_in_range": [0]})
            for o in base.objectives
        ],
        "player_vp": base.player_vp + 5,
        "opponent_vp": base.opponent_vp + 10,
        "player_vp_delta": 5,
        "opponent_vp_delta": 10,
        "objective_control": ["player" for _ in base.objective_control],
        "player_actions": [1 for _ in base.player_models],
        "opponent_actions": [2 for _ in base.opponent_models],
        "player_action_descriptions": ["x" for _ in base.player_models],
        "player_combat_results": [result],
        "opponent_combat_results": [result],
        "player_melee_results": [result],
        "opponent_melee_results": [result],
        "decision": DecisionSnapshot(
            kind="act", model=0, value=3, phase="movement", sub_step=1, episode_step=2
        ),
        "reward": RewardSnapshot(
            total=1.5, breakdown={"x": 1.5}, phase_name="p", phase_index=0
        ),
        "is_terminated": True,
        "is_truncated": True,
        "player_alive_count": 0,
        "opponent_alive_count": 0,
        "player_total_wounds": 0,
        "opponent_total_wounds": 0,
    }
    dynamic = set(GameStateSnapshot.model_fields) - STATIC_SNAPSHOT_FIELDS
    assert set(mutations) == dynamic, sorted(set(mutations) ^ dynamic)
    for field, value in mutations.items():
        if field == "opponent_models" and not base.opponent_models:
            continue
        mutated = base.model_copy(update={field: value})
        rebuilt = apply_delta(base, compute_delta(base, mutated))
        assert rebuilt == mutated, field

    model_mutations: dict[str, object] = {
        "location": [1.0, 2.0],
        "previous_location": [3.0, 4.0],
        "alive": not model.alive,
        "current_wounds": model.current_wounds + 1,
        "advanced_this_turn": not model.advanced_this_turn,
        "charged_this_turn": not model.charged_this_turn,
        "fell_back_this_turn": not model.fell_back_this_turn,
        "distances_to_objectives": [9.0 for _ in model.distances_to_objectives],
        "at_objective": [True for _ in model.at_objective],
        "closest_objective_idx": 7,
        "closest_objective_distance": 7.5,
    }
    assert set(model_mutations) == set(ModelSnapshot.model_fields) - STATIC_MODEL_FIELDS
    for field, value in model_mutations.items():
        changed = model.model_copy(update={field: value})
        mutated = base.model_copy(
            update={"player_models": [changed, *base.player_models[1:]]}
        )
        rebuilt = apply_delta(base, compute_delta(base, mutated))
        assert rebuilt.player_models[0] == changed, field


def test_a_2_7_recording_loads_with_no_decision(recorded_log: EventLog) -> None:
    """The 2.8 field is optional: a log written without it decodes with
    `decision is None` on every snapshot."""
    import json

    lines = JsonMatchCodec().encode(recorded_log).decode().splitlines()
    rewritten = [lines[0]]
    for line in lines[1:]:
        obj = json.loads(line)
        for key in ("snapshot", "anchor", "delta"):
            if isinstance(obj.get(key), dict):
                obj[key].pop("decision", None)
                obj[key].pop("player_melee_results", None)
                obj[key].pop("opponent_melee_results", None)
                if "schema_version" in obj[key]:
                    obj[key]["schema_version"] = "2.7"
        rewritten.append(json.dumps(obj))
    snapshots = ReplayController(
        JsonMatchCodec().decode("\n".join(rewritten).encode())
    ).iter_snapshots()
    assert snapshots
    assert all(s.decision is None for s in snapshots)
    assert all(s.schema_version == "2.7" for s in snapshots)
