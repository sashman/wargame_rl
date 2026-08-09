"""Bit-identical regression gate for the reward pipeline.

The per-step optimisations that follow (memoising `group_cohesion`, hoisting
`closest_objective_v2`'s model-independent candidate mask, sharing the opponent
distance cache) are all meant to be *pure speedups*: same reward, same VP, same
positions, same bits. This file is what makes that a checked claim rather than a
hope.

Assertions use `assert_array_equal`, never `assert_allclose`. These reward values
feed published experiment reports in `reports/`, and a tolerance-based comparison
would wave through exactly the float-reassociation regression a vectorisation
change is most likely to introduce.

The golden `.npz` files are recorded from `main` *before* any optimisation lands.
Regenerate deliberately and never to make a red test go green:

    uv run python -m tests.test_reward_golden --regenerate
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv

DATA_DIR = Path(__file__).parent / "data"
CONFIG_DIR = Path(__file__).parents[1] / "configs"

LAYOUT_SEED = 1234
COMBAT_SEED = 99
ACTION_SEED = 7
N_STEPS = 120

# One config per reward shape worth pinning: the 25v25 control, the config
# actually being trained (its `crowding_exponent` and `observe_objective_control`
# are both new and both read per step), and a curriculum config so phase
# advancement is covered too.
# Paths are relative to `configs/`; the `.npz` key is the stem, so the
# directory prefix does not leak into the fixture filenames.
GOLDEN_CONFIGS = [
    "golden/25v25_single_phase.yaml",
    "golden/25v25_shooting_opponent.yaml",
    "dev/4v4_two_phases.yaml",
]


def _sample_masked_actions(
    mask: np.ndarray | None, rng: np.random.Generator
) -> WargameEnvAction:
    """Pick one valid action per model from an explicitly seeded generator.

    `WargameEnvAction.random` draws from the global `np.random` state, which the
    env itself also perturbs — the trajectory would not be reproducible across
    runs. A local generator keeps the action stream independent of the env's.
    """
    if mask is None:
        raise ValueError("the observation carries no action mask")
    actions = [int(rng.choice(np.flatnonzero(row))) for row in mask]
    return WargameEnvAction(actions=actions)


def _run_trajectory(config_name: str, n_steps: int = N_STEPS) -> dict[str, np.ndarray]:
    """Drive one env deterministically and record everything reward touches."""
    config_path = CONFIG_DIR / config_name
    config = parse_yaml_raw_as(WargameEnvConfig, config_path.read_text())
    env = WargameEnv(config=config)

    observation, _ = env.reset(seed=LAYOUT_SEED, options={"combat_seed": COMBAT_SEED})
    rng = np.random.default_rng(ACTION_SEED)

    rewards: list[float] = []
    per_model: list[np.ndarray] = []
    step_breakdowns: list[dict[str, float]] = []
    player_vp: list[int] = []
    opponent_vp: list[int] = []
    positions: list[np.ndarray] = []

    for _ in range(n_steps):
        action = _sample_masked_actions(observation.action_mask, rng)
        observation, reward, terminated, truncated, _ = env.step(action)

        rewards.append(float(reward))
        per_model.append(np.asarray(env.last_per_model_reward, dtype=np.float64).copy())
        step_breakdowns.append(dict(env.last_reward_breakdown))
        player_vp.append(int(env.player_vp))
        opponent_vp.append(int(env.opponent_vp))
        positions.append(
            np.array(
                [m.location for m in env.wargame_models]
                + [m.location for m in env.opponent_models],
                dtype=np.int64,
            )
        )

        if terminated or truncated:
            observation, _ = env.reset(options={"combat_seed": COMBAT_SEED})

    # Sub-keys come and go legitimately (a calculator only reports
    # `target_switched` on the steps it switches), so pin the union and mark an
    # absent key NaN rather than asserting a fixed key set per step.
    # `assert_array_equal` treats NaN as equal to NaN, so absence is compared too.
    breakdown_keys = sorted({key for step in step_breakdowns for key in step})
    breakdown = np.full((len(step_breakdowns), len(breakdown_keys)), np.nan)
    for row, step in enumerate(step_breakdowns):
        for column, key in enumerate(breakdown_keys):
            if key in step:
                breakdown[row, column] = step[key]

    return {
        "reward": np.array(rewards, dtype=np.float64),
        "per_model_reward": np.stack(per_model),
        "breakdown": breakdown,
        "breakdown_keys": np.array(breakdown_keys),
        "player_vp": np.array(player_vp, dtype=np.int64),
        "opponent_vp": np.array(opponent_vp, dtype=np.int64),
        "positions": np.stack(positions),
    }


def _golden_path(config_name: str) -> Path:
    return DATA_DIR / f"reward_golden_{Path(config_name).stem}.npz"


def regenerate() -> None:
    """Rewrite every golden artefact from the current working tree."""
    DATA_DIR.mkdir(exist_ok=True)
    for config_name in GOLDEN_CONFIGS:
        recorded = _run_trajectory(config_name)
        path = _golden_path(config_name)
        # Keywords spelled out rather than `**recorded`: `savez_compressed`
        # takes `allow_pickle` as its second positional parameter, so a mapping
        # splat is ambiguous to the type checker.
        np.savez_compressed(
            str(path),
            reward=recorded["reward"],
            per_model_reward=recorded["per_model_reward"],
            breakdown=recorded["breakdown"],
            breakdown_keys=recorded["breakdown_keys"],
            player_vp=recorded["player_vp"],
            opponent_vp=recorded["opponent_vp"],
            positions=recorded["positions"],
        )
        print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KiB)")


@pytest.mark.parametrize("config_name", GOLDEN_CONFIGS)
def test_reward_trajectory_is_bit_identical(config_name: str) -> None:
    """Every recorded quantity must match the golden run exactly."""
    path = _golden_path(config_name)
    if not path.exists():
        pytest.skip(f"{path.name} not recorded yet — run with --regenerate")

    golden: Any = np.load(path, allow_pickle=False)
    recorded = _run_trajectory(config_name)

    np.testing.assert_array_equal(
        recorded["breakdown_keys"],
        golden["breakdown_keys"],
        err_msg="the set of active reward calculators changed",
    )
    for key in ("reward", "per_model_reward", "breakdown"):
        np.testing.assert_array_equal(
            recorded[key], golden[key], err_msg=f"{key} diverged from the golden run"
        )
    for key in ("player_vp", "opponent_vp", "positions"):
        np.testing.assert_array_equal(
            recorded[key],
            golden[key],
            err_msg=f"{key} diverged — the change altered game state, not just cost",
        )


if __name__ == "__main__":
    if "--regenerate" not in sys.argv:
        raise SystemExit("pass --regenerate to rewrite the golden artefacts")
    regenerate()
