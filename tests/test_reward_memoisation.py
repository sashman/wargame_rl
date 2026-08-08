"""The two reward hot-path rewrites, proven against the code they replaced.

`tests/test_reward_golden.py` shows that three recorded trajectories are
unchanged. That is necessary but not sufficient: a trajectory only exercises the
states it happens to reach. These tests pin the two rewrites directly —

* `min_distances_to_same_group` was an O(n) Python loop called once per model,
  making `group_cohesion` O(n^3) and 55% of `env.step()`. It is now one
  vectorised pass, memoised per step.
* `closest_objective_v2` rebuilt a `(n_models, n_objectives)` candidate mask
  inside each of its per-model calls, although the mask does not depend on the
  model being scored. It is now built once per step.

— against reference implementations copied from the originals, over inputs a
trajectory would rarely produce (all-dead groups, singleton groups, every
control state).
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from wargame_rl.wargame.envs.env_components.distance_cache import DistanceCache
from wargame_rl.wargame.envs.reward.calculators.closest_objective_v2 import (
    ClosestObjectiveV2Calculator,
)


def _reference_min_distances(
    norms: np.ndarray, group_ids: np.ndarray, alive_mask: np.ndarray | None
) -> np.ndarray:
    """The per-row loop that `min_distances_to_same_group` replaced, verbatim."""
    n = len(group_ids)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        same = (np.arange(n) != i) & (group_ids == group_ids[i])
        if alive_mask is not None:
            same = same & alive_mask
        if not same.any():
            out[i] = 0.0
        else:
            out[i] = float(norms[i, same].min())
    return out


def _make_cache(norms: np.ndarray) -> DistanceCache:
    n = norms.shape[0]
    empty = np.zeros((n, 0))
    return DistanceCache(
        model_obj_deltas=np.zeros((n, 0, 2)),
        model_obj_norms=empty,
        model_obj_norms_offset=empty,
        obj_radii=np.zeros(0),
        model_model_norms=norms,
    )


@settings(max_examples=150, deadline=None)
@given(
    n_models=st.integers(min_value=1, max_value=8),
    n_groups=st.integers(min_value=1, max_value=3),
    seed=st.integers(min_value=0, max_value=2**16),
    use_alive_mask=st.booleans(),
)
def test_vectorised_min_distances_matches_the_loop(
    n_models: int, n_groups: int, seed: int, use_alive_mask: bool
) -> None:
    """Bit-identical, including the all-dead-group and singleton-group cases."""
    rng = np.random.default_rng(seed)
    locations = rng.normal(scale=20.0, size=(n_models, 2))
    deltas = locations[:, None, :] - locations[None, :, :]
    norms = np.linalg.norm(deltas, axis=2, ord=2)
    group_ids = rng.integers(0, n_groups, size=n_models).astype(np.intp)

    alive_mask = None
    if use_alive_mask:
        alive_mask = rng.random(n_models) > 0.4
        # Dead models carry inf in both directions, exactly as `compute_distances`
        # leaves them — the vectorised path must exclude them by mask, not by
        # relying on inf losing the `min`.
        dead = ~alive_mask
        norms = norms.copy()
        norms[dead, :] = np.inf
        norms[:, dead] = np.inf

    expected = _reference_min_distances(norms, group_ids, alive_mask)
    actual = _make_cache(norms).min_distances_to_same_group(
        group_ids, alive_mask=alive_mask
    )

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype


def test_min_distances_alone_in_group_is_zero_not_infinite() -> None:
    """A model with no live group-mate scores 0 — "alone" is not "scattered"."""
    norms = np.array([[0.0, 5.0], [5.0, 0.0]])
    group_ids = np.array([0, 1], dtype=np.intp)

    actual = _make_cache(norms).min_distances_to_same_group(group_ids)

    np.testing.assert_array_equal(actual, np.array([0.0, 0.0]))


def _reference_candidate_mask(
    calculator: ClosestObjectiveV2Calculator,
    player_in_range: np.ndarray,
    player_counts: np.ndarray,
    opponent_counts: np.ndarray,
    model_idx: int,
) -> np.ndarray:
    """The per-model mask construction that `_candidate_mask` replaced, verbatim."""
    n_obj = player_in_range.shape[1]
    model_outside = ~player_in_range[model_idx]
    positive_transition = np.zeros(n_obj, dtype=bool)
    for obj_idx in range(n_obj):
        if not bool(model_outside[obj_idx]):
            continue
        positive_transition[obj_idx] = calculator._is_positive_transition(
            int(player_counts[obj_idx]), int(opponent_counts[obj_idx])
        )
    mask = np.zeros_like(player_in_range, dtype=bool)
    mask[model_idx] = model_outside & positive_transition
    for idx in range(player_in_range.shape[0]):
        if idx == model_idx:
            continue
        idx_outside = ~player_in_range[idx]
        idx_positive = np.zeros(n_obj, dtype=bool)
        for obj_idx in range(n_obj):
            if not bool(idx_outside[obj_idx]):
                continue
            idx_positive[obj_idx] = calculator._is_positive_transition(
                int(player_counts[obj_idx]), int(opponent_counts[obj_idx])
            )
        mask[idx] = idx_outside & idx_positive
    return mask


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_candidate_mask_is_identical_for_every_scored_model(seed: int) -> None:
    """The hoist is only valid if the mask never depended on `model_idx`.

    Asserted for *every* model rather than a sampled one: the whole saving is
    that 25 identical rebuilds became one, so a single-model check would not
    detect the case the optimisation is wrong about.
    """
    rng = np.random.default_rng(seed)
    n_models, n_obj = 6, 4
    player_in_range = rng.random((n_models, n_obj)) > 0.5
    player_counts = player_in_range.sum(axis=0)
    opponent_counts = rng.integers(0, 4, size=n_obj)

    calculator = ClosestObjectiveV2Calculator()
    hoisted = calculator._candidate_mask(
        player_in_range, player_counts, opponent_counts, step_key=(0, 0)
    )

    for model_idx in range(n_models):
        expected = _reference_candidate_mask(
            calculator, player_in_range, player_counts, opponent_counts, model_idx
        )
        np.testing.assert_array_equal(
            hoisted, expected, err_msg=f"mask differs when scoring model {model_idx}"
        )


def test_candidate_mask_covers_every_control_state() -> None:
    """Exhaustive over the count pairs that decide a positive transition.

    `_is_positive_transition` is the only model-independent part of the mask, so
    if it is right for every (player, opponent) count pair the hoist is right
    everywhere — no sampling argument needed.
    """
    calculator = ClosestObjectiveV2Calculator()
    counts = range(0, 6)
    player_counts = np.array([p for p in counts for _ in counts])
    opponent_counts = np.array([o for _ in counts for o in counts])
    n_obj = player_counts.size
    # Every model outside every objective, so the mask is the transition itself.
    player_in_range = np.zeros((3, n_obj), dtype=bool)

    hoisted = calculator._candidate_mask(
        player_in_range, player_counts, opponent_counts, step_key=(0, 0)
    )
    expected = _reference_candidate_mask(
        calculator, player_in_range, player_counts, opponent_counts, model_idx=0
    )

    np.testing.assert_array_equal(hoisted, expected)
