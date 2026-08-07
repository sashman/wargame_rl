"""Model bases occupy space: they stay on the board, and they never overlap."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.types.config import OpponentPolicyConfig
from wargame_rl.wargame.envs.wargame import WargameEnv


def _make_env(n_models: int = 8, n_opponents: int = 4) -> WargameEnv:
    return WargameEnv(
        config=WargameEnvConfig(
            render_mode=None,
            number_of_wargame_models=n_models,
            number_of_opponent_models=n_opponents,
            opponent_policy=OpponentPolicyConfig(type="scripted_advance_to_objective"),
            number_of_objectives=2,
            board_width=40,
            board_height=40,
            number_of_battle_rounds=10,
        )
    )


def _min_gap(env: WargameEnv) -> float:
    """Smallest edge-to-edge gap between any two live bases on the board."""
    models = [m for m in env.wargame_models + env.opponent_models if m.is_alive]
    gap = float("inf")
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            centre = float(np.linalg.norm(a.location - b.location))
            gap = min(gap, centre - a.base_radius - b.base_radius)
    return gap


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(min_value=0, max_value=500))
def test_bases_never_overlap_during_an_episode(seed: int) -> None:
    """No two live bases overlap, at placement or after any step."""
    env = _make_env()
    env.reset(seed=seed)

    assert _min_gap(env) >= -1e-6, "bases overlapped at placement"

    for _ in range(10):
        action = WargameEnvAction(actions=env.action_space.sample())
        _obs, _reward, terminated, truncated, _info = env.step(action)
        assert _min_gap(env) >= -1e-6, "bases overlapped after a step"
        if terminated or truncated:
            break


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(min_value=0, max_value=500))
def test_bases_stay_wholly_on_the_board(seed: int) -> None:
    """A base is part of the model, so no part of it may cross the board edge."""
    env = _make_env()
    env.reset(seed=seed)

    for _ in range(10):
        action = WargameEnvAction(actions=env.action_space.sample())
        _obs, _reward, terminated, truncated, _info = env.step(action)
        for model in env.wargame_models + env.opponent_models:
            radius = model.base_radius
            assert radius <= model.location[0] <= env.board_width - radius
            assert radius <= model.location[1] <= env.board_height - radius
        if terminated or truncated:
            break


def test_no_model_outruns_its_move_allowance() -> None:
    """Collision response never carries a model further than it could have moved.

    Sliding around an obstacle spends movement the model gave up by stopping, so it
    must not become a way to travel further than the Move characteristic allows.
    """
    env = _make_env()
    env.reset(seed=7)
    allowance = env.rules_quantities.max_move_speed

    for _ in range(10):
        before = [m.location.copy() for m in env.wargame_models]
        action = WargameEnvAction(actions=env.action_space.sample())
        _obs, _reward, terminated, truncated, _info = env.step(action)
        for start, model in zip(before, env.wargame_models):
            travelled = float(np.linalg.norm(model.location - start))
            assert travelled <= allowance + 1e-6
        if terminated or truncated:
            break


def test_a_model_is_in_range_of_an_objective_by_its_base_edge() -> None:
    """Range to an objective is measured from the closest part of the base."""
    env = WargameEnv(
        config=WargameEnvConfig(
            render_mode=None,
            number_of_wargame_models=1,
            number_of_objectives=1,
            board_width=20,
            board_height=20,
        )
    )
    env.reset(seed=0)
    radius = env.objectives[0].radius_size
    base = env.wargame_models[0].base_radius
    env.objectives[0].location = np.array([10.0, 10.0])

    # Centre just outside the disc, but the base edge just inside it.
    env.wargame_models[0].location = np.array([10.0 + radius + base / 2, 10.0])
    cache = _cache(env)
    assert bool(cache.model_obj_norms_offset[0, 0] <= cache.obj_radii[0])

    # Push it out by a full base width and it no longer reaches.
    env.wargame_models[0].location = np.array([10.0 + radius + base * 2, 10.0])
    cache = _cache(env)
    assert not bool(cache.model_obj_norms_offset[0, 0] <= cache.obj_radii[0])


def _cache(env: WargameEnv):  # type: ignore[no-untyped-def]
    from wargame_rl.wargame.envs.env_components.distance_cache import compute_distances

    return compute_distances(env.wargame_models, env.objectives)


def test_a_deployment_zone_too_narrow_for_a_base_is_rejected() -> None:
    """A zone that cannot fit a base fails loudly rather than placing it overlapping."""
    env = WargameEnv(
        config=WargameEnvConfig(
            render_mode=None,
            number_of_wargame_models=1,
            number_of_objectives=1,
            board_width=3,
            board_height=3,
        )
    )
    with pytest.raises(RuntimeError, match="too small for a model of radius"):
        env.reset(seed=0)
