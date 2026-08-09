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

import pytest

from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv

LAYOUT_SEED = 42
# 17 is the first action-space seed that reaches a VP-scoring step within 10
# moves on this config; it is the case that was failing intermittently.
SCORING_ACTION_SEED = 17


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
