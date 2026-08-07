"""`objective_hold`'s surplus discount: pay the models that actually hold a point.

Control is a strict count comparison, so an objective needs `opponent_count + 1`
models and every model past that changes nothing about who scores it. Without a
discount the calculator is indifferent between 15 models on one objective and
8/7 across two — both pay `15 x player_value` — while VP is worth double for the
second. The trained batch-3 agent sits at exactly that indifference point:
measured final counts like `player [0,15,0]` against `opp [6,0,14]`.

`surplus_value=1.0` is the default and must stay bit-identical, because every
existing config and checkpoint assumes it.
"""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.envs.env_components.distance_cache import compute_distances
from wargame_rl.wargame.envs.reward.calculators.objective_hold import (
    ObjectiveHoldCalculator,
)
from wargame_rl.wargame.envs.reward.step_context import StepContext
from wargame_rl.wargame.envs.types import (
    ModelConfig,
    ObjectiveConfig,
    OpponentPolicyConfig,
    WargameEnvConfig,
)
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv


def _make_env(n_opponents_on_objective: int) -> WargameEnv:
    """Four player models stacked on objective 0, N opponents contesting it.

    Objective 0 sits at (20, 20) with radius 3; objective 1 is far away and
    empty, so every player model is inside exactly one disc.
    """
    opponents = [
        ModelConfig(x=20, y=20, group_id=0) for _ in range(n_opponents_on_objective)
    ]
    # Any leftover opponents are parked off both objectives so they cannot
    # affect control of either.
    opponents += [ModelConfig(x=2, y=2, group_id=0) for _ in range(4 - len(opponents))]

    config = WargameEnvConfig(
        render_mode=None,
        board_width=40,
        board_height=40,
        number_of_wargame_models=4,
        number_of_opponent_models=4,
        number_of_objectives=2,
        objective_radius_size=3,
        number_of_battle_rounds=6,
        max_groups=1,
        skip_phases=[BattlePhase.command, BattlePhase.charge, BattlePhase.fight],
        # Distinct distances from the centre, so "nearest first" is unambiguous.
        models=[ModelConfig(x=20, y=20 + i, group_id=0) for i in range(4)],
        opponent_models=opponents,
        objectives=[ObjectiveConfig(x=20, y=20), ObjectiveConfig(x=36, y=36)],
        opponent_policy=OpponentPolicyConfig(type="scripted_advance_to_objective"),
    )
    return WargameEnv(config=config)


def _context(env: WargameEnv) -> StepContext:
    """A fresh StepContext, which is also the calculators' per-step cache key."""
    return StepContext(
        distance_cache=compute_distances(env.wargame_models, env.objectives),
        current_turn=1,
        max_turns=env.max_turns,
        board_width=env.board_width,
        board_height=env.board_height,
    )


def _rewards(surplus_value: float, n_opponents: int) -> list[float]:
    """Per-model `objective_hold` values for the four stacked models."""
    env = _make_env(n_opponents)
    env.reset(seed=0)
    calculator = ObjectiveHoldCalculator(weight=1.0, surplus_value=surplus_value)
    context = _context(env)
    return [
        calculator.calculate(i, model, env, context)
        for i, model in enumerate(env.wargame_models)
    ]


def test_default_pays_every_occupant_the_same() -> None:
    """Backward compatibility: 1.0 must not change a single number."""
    rewards = _rewards(surplus_value=1.0, n_opponents=0)

    assert len(set(rewards)) == 1, f"default is no longer uniform: {rewards}"
    assert rewards[0] > 0.0


def test_surplus_models_are_discounted_on_an_uncontested_point() -> None:
    """With no opponents one model holds it; the other three are surplus."""
    rewards = _rewards(surplus_value=0.0, n_opponents=0)

    assert rewards[0] > 0.0, "the nearest model holds the objective"
    assert rewards[1:] == [0.0, 0.0, 0.0], f"surplus was still paid: {rewards}"


def test_the_quota_grows_with_the_opponents_contesting_it() -> None:
    """Two opponents means three models are needed, so only the fourth is surplus.

    This is the property that makes the discount a *reallocation* rather than a
    blanket anti-concentration price: contesting a defended objective still pays
    every model that is actually required to win it.
    """
    rewards = _rewards(surplus_value=0.0, n_opponents=2)

    paid = [r for r in rewards if r > 0.0]
    assert len(paid) == 3, f"expected 3 holders paid, got {rewards}"


@pytest.mark.parametrize("surplus_value", [0.0, 0.25, 0.5])
def test_surplus_never_pays_more_than_a_holder(surplus_value: float) -> None:
    """A discount must not accidentally invert the ordering."""
    rewards = _rewards(surplus_value=surplus_value, n_opponents=0)

    assert rewards[0] >= max(rewards[1:])


def test_quota_is_recomputed_each_step_not_frozen_after_the_first() -> None:
    """The two caches use separate keys; sharing one froze the quota at step 1.

    `_objective_values` stamps `_cached_ctx` before the quota is ever built, so
    a shared key would make every later step reuse step one's holders.
    """
    env = _make_env(0)
    env.reset(seed=0)
    calculator = ObjectiveHoldCalculator(weight=1.0, surplus_value=0.0)

    first_context = _context(env)
    calculator.calculate(0, env.wargame_models[0], env, first_context)
    quota_after_first = calculator._cached_within_quota

    second_context = _context(env)
    calculator.calculate(0, env.wargame_models[0], env, second_context)

    assert calculator._cached_quota_ctx is second_context
    assert quota_after_first is not calculator._cached_within_quota
    assert isinstance(calculator._cached_within_quota, np.ndarray)
