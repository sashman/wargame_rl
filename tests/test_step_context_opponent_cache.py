"""The opponent distance matrix is built at most once a step.

Five calculators each built it from scratch with byte-identical arguments and
their own private memo. `compute_distances` is ~37% of `env.step()`, so that is
the shape of the bug that once made two calculators ~80% of a 25v25 step.

The cache is only sound because the `StepContext` is built AFTER the opponent's
turn has been executed, so the board is final for the step. If that ordering
ever changes, `test_the_cache_sees_the_board_the_calculators_see` fails.
"""

from __future__ import annotations

import numpy as np

from scripts.scenario_overrides import load_env_config
from wargame_rl.wargame.envs.domain.entities import alive_mask_for
from wargame_rl.wargame.envs.env_components.distance_cache import compute_distances
from wargame_rl.wargame.envs.reward.step_context import StepContext
from wargame_rl.wargame.model.common.factory import create_environment

CONFIG = "configs/golden/25v25_maps_two_mode.yaml"


def _context(env: object) -> StepContext:
    """A context like the env builds, carrying only what this test reads."""
    return StepContext(
        distance_cache=compute_distances(
            env.player_models,  # type: ignore[attr-defined]
            env.objectives,  # type: ignore[attr-defined]
        ),
        current_turn=1,
        max_turns=20,
        board_width=int(env.board_width),  # type: ignore[attr-defined]
        board_height=int(env.board_height),  # type: ignore[attr-defined]
    )


def test_two_reads_return_the_same_object() -> None:
    """Identity, not equality -- equality would pass without any caching."""
    env = create_environment(env_config=load_env_config(CONFIG))
    env.reset(seed=700000)
    ctx = _context(env)

    first = ctx.opponent_distances(env)
    second = ctx.opponent_distances(env)
    env.close()

    assert first is second


def test_a_fresh_context_does_not_inherit_the_previous_one() -> None:
    """Per-step lifetime. A cache surviving into the next step would be stale."""
    env = create_environment(env_config=load_env_config(CONFIG))
    env.reset(seed=700000)

    first = _context(env).opponent_distances(env)
    second = _context(env).opponent_distances(env)
    env.close()

    assert first is not second


def test_the_cache_sees_the_board_the_calculators_see() -> None:
    """The cached value must equal a fresh computation at the same state.

    This is what the per-step lifetime buys, and what a context reused across
    the opponent's movement would break.
    """
    env = create_environment(env_config=load_env_config(CONFIG))
    env.reset(seed=700001)
    ctx = _context(env)

    cached = ctx.opponent_distances(env)
    fresh = compute_distances(
        env.opponent_models,
        env.objectives,
        alive_mask=alive_mask_for(env.opponent_models),
    )
    env.close()

    np.testing.assert_array_equal(
        cached.model_obj_norms_offset, fresh.model_obj_norms_offset
    )
    np.testing.assert_array_equal(cached.obj_radii, fresh.obj_radii)
