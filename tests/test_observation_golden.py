"""Bit-identical regression gate for the observation -> tensor pipeline.

`tests/test_reward_golden.py` pins everything the *reward* pipeline produces, but
it does not pin the network's input. The action mask is covered indirectly —
actions are drawn from it, so a mask change moves `positions` — and nothing else
is. A wrong per-model feature column therefore passes the entire suite while
silently degrading learning, which is precisely the failure the planned
observation-encoder work risks: 38 of the 49 per-model columns are constant for a
run and are candidates for hoisting out of the per-step path.

Two distinct things are checked here, and the second matters more than the first:

1. The six arrays are bit-identical to a recorded trajectory (`assert_array_equal`,
   never `assert_allclose`, for the same reason the reward gate says so).
2. The **alive column is still where the network looks for it**. That is not
   implied by (1): a change that reorders columns *and* is regenerated into the
   golden file would pass (1) forever. `TransformerNetwork._alive_feature_index`
   counts backwards from the last column, and `_alive_from_features` degrades to
   "everything is alive" when the index falls out of range rather than raising —
   so the key-padding mask starts attending dead models and nothing anywhere says
   so. This is documented in `model/CLAUDE.md` as the trap that bit the last
   feature added here.

Regenerate deliberately and never to make a red test go green:

    uv run python -m tests.test_observation_golden --regenerate
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.observation import (
    _observation_to_numpy,
    observation_to_tensor,
    observations_to_tensor_batch,
)
from wargame_rl.wargame.model.net import TransformerNetwork

DATA_DIR = Path(__file__).parent / "data"
CONFIG_DIR = Path(__file__).parents[1] / "examples" / "env_config"

# Deliberately the same seeds as `test_reward_golden`, so a divergence can be
# read across both files as the same trajectory.
LAYOUT_SEED = 1234
COMBAT_SEED = 99
ACTION_SEED = 7
N_STEPS = 60

GOLDEN_CONFIGS = [
    "25v25_single_phase.yaml",
    "25v25_shooting_opponent.yaml",
    "4v4_scripted_opponent_fixed_objectives_2_reward_phases.yaml",
]

ARRAY_NAMES = (
    "game",
    "objectives",
    "player_models",
    "opponent_models",
    "terrain",
    "action_mask",
)


def _load_env(config_name: str) -> WargameEnv:
    config = parse_yaml_raw_as(WargameEnvConfig, (CONFIG_DIR / config_name).read_text())
    return WargameEnv(config=config)


def _sample_masked_actions(
    mask: np.ndarray | None, rng: np.random.Generator
) -> WargameEnvAction:
    """Pick one valid action per model from an explicitly seeded generator."""
    if mask is None:
        raise ValueError("the observation carries no action mask")
    return WargameEnvAction(
        actions=[int(rng.choice(np.flatnonzero(row))) for row in mask]
    )


def _run_trajectory(config_name: str, n_steps: int = N_STEPS) -> dict[str, np.ndarray]:
    """Record the six observation arrays at every step of a fixed trajectory."""
    env = _load_env(config_name)
    observation, _ = env.reset(seed=LAYOUT_SEED, options={"combat_seed": COMBAT_SEED})
    rng = np.random.default_rng(ACTION_SEED)

    collected: dict[str, list[np.ndarray]] = {name: [] for name in ARRAY_NAMES}

    for _ in range(n_steps):
        arrays = _observation_to_numpy(observation)
        for name, array in zip(ARRAY_NAMES, arrays):
            # `action_mask` is the only one that may be None; represent its
            # absence as an empty array so the shape difference is still caught.
            collected[name].append(
                np.zeros((0, 0)) if array is None else np.asarray(array)
            )

        action = _sample_masked_actions(observation.action_mask, rng)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            observation, _ = env.reset(options={"combat_seed": COMBAT_SEED})

    return {name: np.stack(values) for name, values in collected.items()}


def _golden_path(config_name: str) -> Path:
    return DATA_DIR / f"observation_golden_{Path(config_name).stem}.npz"


def regenerate() -> None:
    """Rewrite every golden artefact from the current working tree."""
    DATA_DIR.mkdir(exist_ok=True)
    for config_name in GOLDEN_CONFIGS:
        recorded = _run_trajectory(config_name)
        path = _golden_path(config_name)
        np.savez_compressed(
            str(path),
            game=recorded["game"],
            objectives=recorded["objectives"],
            player_models=recorded["player_models"],
            opponent_models=recorded["opponent_models"],
            terrain=recorded["terrain"],
            action_mask=recorded["action_mask"],
        )
        print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KiB)")


@pytest.mark.parametrize("config_name", GOLDEN_CONFIGS)
def test_observation_arrays_are_bit_identical(config_name: str) -> None:
    """Every observation array must match the golden run exactly."""
    path = _golden_path(config_name)
    if not path.exists():
        pytest.skip(f"{path.name} not recorded yet — run with --regenerate")

    golden: Any = np.load(path, allow_pickle=False)
    recorded = _run_trajectory(config_name)

    for name in ARRAY_NAMES:
        np.testing.assert_array_equal(
            recorded[name],
            golden[name],
            err_msg=(
                f"{name} diverged from the golden observation run — the network "
                "is being fed something different"
            ),
        )


@pytest.mark.parametrize("config_name", GOLDEN_CONFIGS)
def test_alive_column_is_where_the_network_looks(config_name: str) -> None:
    """What the network believes is alive must equal what the env says.

    The end-to-end form of the column-order trap, and the reason it is not
    redundant with the golden arrays: this assertion is anchored to the env's own
    truth, so it survives a deliberate regeneration of the golden files. A column
    reorder that was regenerated into the goldens would pass every other test in
    this file and fail only here.

    The casualties are inflicted directly rather than waited for. Under random
    actions only the shooting config reliably kills anything inside a short
    trajectory, and an all-alive board proves nothing: `_alive_from_features`
    *also* degrades to all-alive when the index falls out of range, so both the
    right and the wrong answer look identical. Killing a deliberately asymmetric
    subset means a wrong column cannot coincidentally agree either.

    One model is also left **wounded but alive**, and that is what gives the test
    its teeth. The column immediately after `alive` is `wound_ratio`, and every
    config here runs `max_wounds: 1` — where `wound_ratio` is 1.0 or 0.0 and so
    is *bit-identical to `alive`*. An off-by-one column read is therefore
    undetectable on this scenario until some model has partial wounds. Verified:
    without the wounded model, appending a per-model column (the exact documented
    trap) passes this test.
    """
    env = _load_env(config_name)
    observation, _ = env.reset(seed=LAYOUT_SEED, options={"combat_seed": COMBAT_SEED})
    rng = np.random.default_rng(ACTION_SEED)
    # Pinned to CPU: this is a layout assertion with no arithmetic in it, and
    # both helpers otherwise default to `auto_device()` — which would make the
    # test unrunnable on the CI box that has no GPU.
    network = TransformerNetwork.policy_from_env(env).to("cpu")

    n_models = len(env.wargame_models)
    assert n_models >= 4, f"{config_name} is too small to make this test meaningful"
    dead_indices = {0, n_models // 2, n_models - 1}
    wounded_index = 1

    def apply_casualties() -> None:
        for index in dead_indices:
            env.wargame_models[index].take_damage(10_000)
        # Alive, but with a wound ratio below the 0.5 threshold the network
        # applies — so `alive` and `wound_ratio` finally disagree on this row.
        wounded = env.wargame_models[wounded_index]
        wounded.stats["max_wounds"] = 4
        wounded.stats["current_wounds"] = 1

    # The condition that makes this test discriminating has to be observed, not
    # assumed: on the shooting config the wounded model can itself be killed
    # before the run ends, so a terminal assertion about its state is flaky.
    saw_discriminating_state = False

    for step in range(N_STEPS):
        # Applied partway in, so the all-alive layout is checked too.
        if step == N_STEPS // 2:
            apply_casualties()
            observation, _, _, _, _ = env.step(
                _sample_masked_actions(observation.action_mask, rng)
            )

        saw_discriminating_state = saw_discriminating_state or any(
            m.is_alive and m.stats["current_wounds"] * 2 < m.stats["max_wounds"]
            for m in env.wargame_models
        )
        tensors = observation_to_tensor(observation, device="cpu")
        player_tensor = tensors[2].unsqueeze(0)
        n_opponents = int(tensors[3].shape[0])

        believed = network._alive_from_features(player_tensor, n_opponents)[0]
        truth = torch.tensor([m.is_alive for m in env.wargame_models], dtype=torch.bool)

        assert torch.equal(believed, truth), (
            f"the network read the wrong column as `alive` on {config_name} at "
            f"step {step}: believed {int(believed.sum())} alive, env has "
            f"{int(truth.sum())}. A per-model feature column was added or "
            "reordered — `alive` must stay inside `core`, never appended after "
            "the combat stats."
        )

        action = _sample_masked_actions(observation.action_mask, rng)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            # A reset revives everything, so re-apply the casualties.
            observation, _ = env.reset(options={"combat_seed": COMBAT_SEED})
            apply_casualties()
            observation, _, _, _, _ = env.step(
                _sample_masked_actions(observation.action_mask, rng)
            )

    assert not all(m.is_alive for m in env.wargame_models), (
        "the trajectory ended with every model alive, so the assertion above "
        "could not distinguish a correct column from an out-of-range one"
    )
    assert saw_discriminating_state, (
        "no model was ever alive with a wound ratio below the network's 0.5 "
        "threshold, so `alive` and `wound_ratio` agreed on every row and an "
        "off-by-one column read would have gone undetected"
    )


@pytest.mark.parametrize("config_name", GOLDEN_CONFIGS)
def test_batch_conversion_matches_the_single_path(config_name: str) -> None:
    """`observations_to_tensor_batch` must agree with stacking one at a time.

    Training reads the batch path exclusively, while every test that inspects an
    observation reads the single path — so a divergence between them would be
    invisible to both. Any preallocated-buffer rewrite makes this a live risk,
    since the two would then be genuinely different code.
    """
    env = _load_env(config_name)
    observation, _ = env.reset(seed=LAYOUT_SEED, options={"combat_seed": COMBAT_SEED})
    rng = np.random.default_rng(ACTION_SEED)

    observations = []
    for _ in range(8):
        observations.append(observation)
        action = _sample_masked_actions(observation.action_mask, rng)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            observation, _ = env.reset(options={"combat_seed": COMBAT_SEED})

    batched = observations_to_tensor_batch(observations, device="cpu")
    singles = [observation_to_tensor(obs, device="cpu") for obs in observations]

    for index, name in enumerate(ARRAY_NAMES):
        stacked = torch.stack([single[index] for single in singles])
        torch.testing.assert_close(
            batched[index],
            stacked,
            rtol=0,
            atol=0,
            msg=f"{name} differs between the batch and single conversion paths",
        )


if __name__ == "__main__":
    if "--regenerate" not in sys.argv:
        raise SystemExit("pass --regenerate to rewrite the golden artefacts")
    regenerate()
