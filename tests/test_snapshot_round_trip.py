"""`load_state` must restore everything `to_snapshot` records.

Found via a flake: `test_v9_milestone_validation.py` asserts
`snap == env.load_state(snap).to_snapshot()` but drives the env with an
**unseeded** `action_space.sample()`, so it explores a different trajectory on
every run and only reached a diverging state about 1 run in 40.

The divergence was real. `load_state` hardcoded both VP deltas to 0 while
`to_snapshot` wrote them out faithfully, so any state whose capture step scored
VP round-tripped to a different snapshot. `player_vp_delta` is a feature of the
observation the policy acts on (game features, index 5), so a replayed or
injected state fed the network an input the live env would never have produced.

These tests sweep the action-space seed deliberately rather than leaving it to
chance, so the states that expose it are reached every run.
"""

from __future__ import annotations

import json

import pytest

from wargame_rl.wargame.envs.state.codecs import JsonMatchCodec
from wargame_rl.wargame.envs.state.event_log import EventLog
from wargame_rl.wargame.envs.state.replay import ReplayController
from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.types.config import TerrainPieceConfig
from wargame_rl.wargame.envs.wargame import WargameEnv

LAYOUT_SEED = 42
# 10 is the first action-space seed that reaches a VP-scoring step within 10
# moves on this config; it is the case that was failing intermittently. It was
# 17 before the board became continuous -- which seed scores is a property of
# the dynamics, and the assertion at the top of the test is what stops a
# re-pick from quietly making this vacuous.
SCORING_ACTION_SEED = 10


def _env() -> WargameEnv:
    return WargameEnv(
        config=WargameEnvConfig(
            board_width=20,
            board_height=20,
            number_of_wargame_models=3,
            number_of_objectives=2,
            number_of_battle_rounds=5,
        )
    )


def _trajectory(action_seed: int, n_steps: int = 10) -> tuple[WargameEnv, list]:
    env = _env()
    env.reset(seed=LAYOUT_SEED)
    env.action_space.seed(action_seed)

    snapshots = [env.to_snapshot()]
    for _ in range(n_steps):
        _, _, terminated, truncated, _ = env.step(
            WargameEnvAction(actions=env.action_space.sample())
        )
        snapshots.append(env.to_snapshot())
        if terminated or truncated:
            break
    return env, snapshots


@pytest.mark.parametrize("action_seed", range(20))
def test_load_state_round_trips_every_snapshot(action_seed: int) -> None:
    """Restoring any snapshot and re-capturing must return the same snapshot."""
    env, snapshots = _trajectory(action_seed)

    for snapshot in snapshots:
        env.load_state(snapshot)
        assert env.to_snapshot() == snapshot


def test_the_vp_deltas_specifically_survive_the_round_trip() -> None:
    """The field that was being zeroed, pinned on a state that actually scores.

    Asserts a non-zero delta is present first — otherwise the round trip would
    be comparing 0 to 0 and would pass against the very bug it exists to catch.
    """
    env, snapshots = _trajectory(SCORING_ACTION_SEED)

    scoring = [s for s in snapshots if s.player_vp_delta or s.opponent_vp_delta]
    assert scoring, (
        "no step scored VP on this seed, so the test cannot distinguish a "
        "restored delta from a zeroed one — pick another action seed"
    )

    for snapshot in scoring:
        env.load_state(snapshot)
        restored = env.to_snapshot()
        assert restored.player_vp_delta == snapshot.player_vp_delta
        assert restored.opponent_vp_delta == snapshot.opponent_vp_delta


# ---------------------------------------------------------------------------
# Terrain in recordings (schema 2.1): static terrain is stored on the reset and
# every anchor, never in a delta, so a replay reconstructs it at any step.
# ---------------------------------------------------------------------------

_ANCHOR_INTERVAL = 3


def _terrain_recording() -> tuple[EventLog, list[list[list[float]]]]:
    """A short recorded episode on a fixed-terrain config, plus its footprints."""
    env = WargameEnv(
        config=WargameEnvConfig(
            board_width=20,
            board_height=20,
            number_of_wargame_models=2,
            number_of_objectives=1,
            number_of_battle_rounds=8,
            terrain=[
                TerrainPieceConfig(footprint=(5, 5, 8, 8)),
                TerrainPieceConfig(footprint=(12, 12, 15, 16)),
            ],
        )
    )
    env.reset(seed=LAYOUT_SEED)
    env.action_space.seed(0)
    expected = env.to_snapshot().terrain_footprints
    assert expected is not None

    log = EventLog(anchor_interval=_ANCHOR_INTERVAL)
    log.record_reset(env.to_snapshot())
    for _ in range(9):
        env.step(WargameEnvAction(actions=env.action_space.sample()))
        log.record_step(env.to_snapshot())
    return log, expected


def test_terrain_survives_encode_decode() -> None:
    """Every reconstructed frame carries the recorded terrain outlines."""
    log, expected = _terrain_recording()

    decoded = JsonMatchCodec().decode(JsonMatchCodec().encode(log))
    snapshots = ReplayController(decoded).iter_snapshots()

    assert len(snapshots) > _ANCHOR_INTERVAL  # crosses at least one anchor
    for snapshot in snapshots:
        assert snapshot.terrain_footprints == expected


def test_terrain_survives_a_mid_anchor_seek() -> None:
    """A seek that resumes from a mid-episode anchor still has terrain.

    Terrain lives on anchors, not deltas, so a `seek` that rebuilds from the
    nearest anchor forward would drop it if anchors did not carry it.
    """
    log, expected = _terrain_recording()
    controller = ReplayController(JsonMatchCodec().decode(JsonMatchCodec().encode(log)))

    past_first_anchor = _ANCHOR_INTERVAL + 2
    snapshot = controller.seek(past_first_anchor)

    assert snapshot.step == past_first_anchor
    assert snapshot.terrain_footprints == expected


def test_pre_2_1_recording_decodes_without_terrain() -> None:
    """A recording written before schema 2.1 loads with terrain_footprints=None."""
    log, _ = _terrain_recording()
    data = JsonMatchCodec().encode(log)

    # Simulate an old recording: drop the field and revert the version everywhere.
    def _scrub(value: object) -> None:
        if isinstance(value, dict):
            value.pop("terrain_footprints", None)
            if value.get("schema_version") == "2.1":
                value["schema_version"] = "2.0"
            for child in value.values():
                _scrub(child)
        elif isinstance(value, list):
            for child in value:
                _scrub(child)

    lines = data.decode().splitlines()
    rewritten = [lines[0]]
    for line in lines[1:]:
        obj = json.loads(line)
        _scrub(obj)
        rewritten.append(json.dumps(obj))
    old_data = "\n".join(rewritten).encode()

    snapshots = ReplayController(JsonMatchCodec().decode(old_data)).iter_snapshots()

    assert snapshots  # decoded without raising
    assert all(s.schema_version == "2.0" for s in snapshots)
    assert all(s.terrain_footprints is None for s in snapshots)
