"""Recording the per-model facade to a match event log, at both cadences.

At `phase` cadence the log is the phase facade's: one snapshot per settled
reward window, `step` the phase counter, so every reader works unchanged. At
`decision` cadence there is one snapshot per `step()` carrying the decision,
with its own counter and budget so anchoring and seeking still hold.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.per_model_seats import small_config
from tests.test_event_stream import assert_snapshots_agree
from wargame_rl.wargame.envs.baseline.registry import build_baseline_policy
from wargame_rl.wargame.envs.debug.reproduce import provenance_of
from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.envs.per_model import (
    PerModelAction,
    PerModelEnv,
    StepKind,
    random_chooser,
    record_episode,
    scripted_chooser,
)
from wargame_rl.wargame.envs.per_model.random_seat import random_legal_action
from wargame_rl.wargame.envs.state import (
    EpisodeProvenance,
    EventLogExporter,
    GameStateSnapshot,
    JsonMatchCodec,
    PerModelProvenance,
    ReplayController,
    StepEvent,
    analyze_match,
    cadence_of,
    facade_of,
    validate_snapshot,
)
from wargame_rl.wargame.envs.state.snapshot import ModelSnapshot
from wargame_rl.wargame.envs.types import BattlePhase, WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv

SEED = 500010


def _play(env: PerModelEnv, seed: int, *, policy: str | None = None) -> None:
    """One whole episode, scripted when `policy` is given, else random legal."""
    if policy is not None:
        choose = scripted_chooser(build_baseline_policy(policy), [env])
    else:
        choose = random_chooser(seed)
    observation, _ = env.reset(seed=seed)
    done = False
    while not done:
        action = choose([env], [observation])[0]
        observation, _, done, _, _ = env.step(action)


def _recorded(
    config: WargameEnvConfig, cadence: str, seed: int = SEED, anchor: int = 10
) -> tuple[PerModelEnv, EventLogExporter]:
    exporter = EventLogExporter(anchor_interval=anchor)
    env = PerModelEnv(config, state_exporters=[exporter], record_cadence=cadence)  # type: ignore[arg-type]
    _play(env, seed, policy="squad_march_take")
    return env, exporter


# --------------------------------------------------------------- phase cadence


def test_phase_cadence_records_one_snapshot_per_phase_step() -> None:
    env, exporter = _recorded(small_config(opponent_x=22), "phase")
    snapshots = ReplayController(exporter.log).iter_snapshots()
    assert [s.step for s in snapshots] == list(range(env.max_turns + 1))
    assert snapshots[-1].is_terminated and snapshots[-1].step == snapshots[-1].max_steps
    assert all(s.decision is None for s in snapshots)
    assert all(validate_snapshot(s, env.config) == [] for s in snapshots)


def test_a_stepped_command_phase_settles_two_windows_from_one_close() -> None:
    """A `close_turn` settles the window it closes and, when command is
    stepped, the command window `_advance` opens and settles at once: two
    events from one `step()`, consecutive phase steps."""
    config = small_config(
        opponent_x=22,
        skip_phases=[
            BattlePhase.charge,
            BattlePhase.pile_in,
            BattlePhase.fight,
            BattlePhase.consolidate,
        ],
    )
    env, exporter = _recorded(config, "phase")
    steps = [e.delta.step for e in exporter.log.events if isinstance(e, StepEvent)]
    assert steps == list(range(1, env.max_turns + 1))
    assert env.stepped_phases_per_round == 3


def test_the_window_snapshot_carries_the_acts_and_the_volley() -> None:
    env = PerModelEnv(small_config(opponent_x=22), state_exporters=[EventLogExporter()])
    observation, _ = env.reset(seed=SEED)
    # Hand-step the first movement window: one act, then close the unit and
    # the turn through random legal decisions.
    acted: dict[int, int] = {}
    done = False
    while not done and env.game_clock_state.phase is BattlePhase.movement:
        point = observation.decision
        action = random_legal_action(point, np.random.default_rng(1))
        if action.kind is StepKind.act:
            acted[action.model] = action.value
        observation, _, done, _, info = env.step(action)
        if info["reward_settled"]:
            break
    snapshots = ReplayController(env.state_exporters[0].log).iter_snapshots()  # type: ignore[attr-defined]
    movement = [s for s in snapshots if s.action_phase == "movement"]
    assert movement, "no movement window settled"
    recorded = movement[-1].player_actions
    assert recorded is not None
    for index, value in acted.items():
        assert recorded[index] == value
    assert all(
        recorded[i] == STAY_ACTION for i in range(len(recorded)) if i not in acted
    )


def test_the_shooting_window_carries_its_volley_and_the_next_does_not() -> None:
    env, exporter = _recorded(
        small_config(opponent_x=22, baseline="squad_march_shoot"), "phase"
    )
    snapshots = ReplayController(exporter.log).iter_snapshots()
    shooting = [s for s in snapshots if s.action_phase == "shooting"]
    following = [
        snapshots[i + 1]
        for i, s in enumerate(snapshots[:-1])
        if s.action_phase == "shooting" and snapshots[i + 1].action_phase == "movement"
    ]
    assert any(s.player_combat_results for s in shooting)
    assert all(not s.player_combat_results for s in following)


def test_attaching_an_exporter_changes_nothing_settled() -> None:
    config = small_config(opponent_x=22)
    plain = PerModelEnv(config)
    recorded = PerModelEnv(config, state_exporters=[EventLogExporter()])
    for env in (plain, recorded):
        choose = scripted_chooser(build_baseline_policy("squad_march_take"), [env])
        observation, info = env.reset(seed=SEED, options={"combat_seed": 7})
        settled = [w.state for w in info["settled"]]
        done = False
        while not done:
            observation, _, done, _, info = env.step(choose([env], [observation])[0])
            settled.extend(w.reward for w in info["settled"])
        env.__dict__["_settled_trace"] = settled
    assert plain.__dict__["_settled_trace"] == recorded.__dict__["_settled_trace"]
    assert plain.player_vp == recorded.player_vp


# ------------------------------------------------------------ decision cadence


def _blank_dead(snapshot: GameStateSnapshot) -> GameStateSnapshot:
    def blank(models: list[ModelSnapshot]) -> list[ModelSnapshot]:
        return [
            m
            if m.alive
            else m.model_copy(
                update={
                    "closest_objective_idx": None,
                    "closest_objective_distance": None,
                }
            )
            for m in models
        ]

    blanked: GameStateSnapshot = snapshot.model_copy(
        update={
            "player_models": blank(snapshot.player_models),
            "opponent_models": blank(snapshot.opponent_models),
        }
    )
    return blanked


def test_decision_cadence_records_one_snapshot_per_step() -> None:
    config = small_config(opponent_x=22)
    exporter = EventLogExporter(anchor_interval=5)
    env = PerModelEnv(config, state_exporters=[exporter], record_cadence="decision")
    rng = np.random.default_rng(3)
    observation, _ = env.reset(seed=SEED)
    live = [env.to_snapshot()]
    actions: list[PerModelAction] = []
    done = False
    while not done:
        action = random_legal_action(observation.decision, rng)
        actions.append(action)
        observation, _, done, _, info = env.step(action)
        live.append(env.to_snapshot())
    controller = ReplayController(
        JsonMatchCodec().decode(JsonMatchCodec().encode(exporter.log))
    )
    snapshots = controller.iter_snapshots()
    assert len(snapshots) == len(actions) + 1 == env.episode_step + 1
    assert [s.step for s in snapshots] == list(range(env.episode_step + 1))
    for snapshot, action in zip(snapshots[1:], actions):
        assert snapshot.decision is not None
        assert snapshot.decision.kind == action.kind.value
        if action.kind is StepKind.close_turn:
            assert snapshot.decision.model is None and snapshot.decision.value is None
        else:
            assert (snapshot.decision.model, snapshot.decision.value) == (
                action.model,
                action.value,
            )
    assert snapshots[-1].is_terminated
    assert all(validate_snapshot(s, config) == [] for s in snapshots)
    assert env.episode_step <= env.decision_budget
    # Seeking rebuilds every step from the nearest anchor, exactly -- except
    # a dead model's `closest_objective_*`, which a delta cannot set back to
    # None (None means "unchanged" in `ModelDelta`); a pre-existing codec
    # limit on both facades, so the comparison blanks them on both sides.
    for step, expected in enumerate(live):
        assert_snapshots_agree(
            _blank_dead(controller.seek(step)), _blank_dead(expected)
        )
    anchors = [
        e.delta.step
        for e in exporter.log.events
        if isinstance(e, StepEvent) and e.anchor
    ]
    assert anchors and all(step % 5 == 0 for step in anchors)


@pytest.mark.parametrize("seed", [500011, 500012])
def test_the_decision_budget_bounds_every_random_episode(seed: int) -> None:
    for melee in (False, True):
        env = PerModelEnv(small_config(opponent_x=22, melee=melee, skip_phases=[]))
        _play(env, seed)
        assert env.episode_step <= env.decision_budget


# ----------------------------------------------------------------- provenance


def test_provenance_round_trips_the_facade_and_the_cadence() -> None:
    _env, exporter = _recorded(small_config(opponent_x=22), "decision")
    assert isinstance(exporter.log.provenance, PerModelProvenance)
    decoded = JsonMatchCodec().decode(JsonMatchCodec().encode(exporter.log))
    assert isinstance(decoded.provenance, PerModelProvenance)
    assert facade_of(decoded.provenance) == "per_model"
    assert cadence_of(decoded.provenance) == "decision"


def test_a_phase_facade_recording_is_still_the_phase_facades() -> None:
    exporter = EventLogExporter()
    env = WargameEnv(small_config(opponent_x=22), state_exporters=[exporter])
    env.reset(seed=SEED)
    decoded = JsonMatchCodec().decode(JsonMatchCodec().encode(exporter.log))
    assert type(decoded.provenance) is EpisodeProvenance
    assert facade_of(decoded.provenance) == "phase"
    assert cadence_of(decoded.provenance) == "phase"


def test_an_unknown_facade_tag_is_refused_rather_than_dropped() -> None:
    """Before `decode_provenance`, the lenient parent swallowed the tag and
    read a per-model recording as the phase facade's."""
    _env, exporter = _recorded(small_config(opponent_x=22), "phase")
    lines = JsonMatchCodec().encode(exporter.log).decode().splitlines()
    header = json.loads(lines[0])
    header["provenance"]["facade"] = "other"
    lines[0] = json.dumps(header)
    with pytest.raises(ValueError, match="unknown facade 'other'"):
        JsonMatchCodec().decode("\n".join(lines).encode())


# ------------------------------------------------------------------- readers


def test_analyze_match_refuses_decision_cadence_and_accepts_phase() -> None:
    config = small_config(opponent_x=22)
    _env, phase = _recorded(config, "phase")
    analysis = analyze_match(
        ReplayController(phase.log).iter_snapshots(), "phase.jsonl"
    )
    assert analysis.file == "phase.jsonl"
    _env, decision = _recorded(config, "decision")
    with pytest.raises(ValueError, match="decision cadence"):
        analyze_match(ReplayController(decision.log).iter_snapshots(), "decision.jsonl")


def test_debug_reproduction_refuses_a_per_model_recording() -> None:
    _env, exporter = _recorded(small_config(opponent_x=22), "phase")
    with pytest.raises(ValueError, match="per_model"):
        provenance_of(exporter.log, "per_model.jsonl")


def test_record_episode_writes_a_decodable_log_with_the_driver(tmp_path: Path) -> None:
    config = small_config(opponent_x=22)
    out = record_episode(
        lambda envs: scripted_chooser(build_baseline_policy("squad_march_take"), envs),
        config,
        SEED,
        tmp_path / "log.jsonl",
        driver="squad_march_take",
    )
    log = JsonMatchCodec().decode(out.read_bytes())
    assert log.provenance is not None and log.provenance.driver == "squad_march_take"
    snapshots = ReplayController(log).iter_snapshots()
    assert snapshots[-1].is_terminated
