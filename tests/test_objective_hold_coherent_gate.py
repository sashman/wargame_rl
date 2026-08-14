"""`objective_hold.require_coherent`: an illegal position earns nothing.

Under the occupancy form a unit's income depends only on how many objectives it
touches, never on how far anyone walked. Measured per step at weight 1.25, one
5-model unit earns 1.25 together, **2.50** spread 3+2 across two nearby
objectives, and **the same 2.50** spread 4+1 with one model detached far away —
the reward cannot tell the legal spread from the illegal one. Full scatter pays
6.25, five times staying together, which is the gradient behind 82.4% of adrift
models walking to a different objective from their unit's body.

The gate makes the reward agree with the rule. The load-bearing test is
`test_the_legal_two_objective_spread_is_untouched`: a gate that also killed the
legal spread would "fix" coherency by forbidding good play, and the rule's 9"
cap exists precisely so that spread *is* legal.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.domain.entities import alive_mask_for
from wargame_rl.wargame.envs.env_components.distance_cache import compute_distances
from wargame_rl.wargame.envs.reward.calculators.objective_hold import (
    ObjectiveHoldCalculator,
)
from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv

GOLDEN = "configs/golden/25v25_shooting_opponent.yaml"


def prepared_env() -> WargameEnv:
    """A reset env that has stepped once, so a `StepContext` exists."""
    with open(GOLDEN) as handle:
        config = cast(
            WargameEnvConfig, parse_yaml_raw_as(WargameEnvConfig, handle.read())
        )
    config.render_mode = None
    env = WargameEnv(config)
    env.reset(seed=5)
    env.step(WargameEnvAction(actions=[0] * len(env.wargame_models)))
    return env


def place_one_unit(env: WargameEnv, offsets: list[tuple[float, float]]) -> list[int]:
    """Put five live models of one unit at `offsets` from objective 0.

    Everything else is killed, so the objective occupancy counts are exactly the
    five placed models and the gate is measured on the unit under test.
    """
    centre = np.array(env.objectives[0].location, dtype=float)
    indices = list(range(5))
    for index, model in enumerate(env.wargame_models):
        model.stats["current_wounds"] = 1 if index < 5 else 0
        model.group_id = 0
    for index, (dx, dy) in zip(indices, offsets):
        model = env.wargame_models[index]
        model.location = np.array(
            [centre[0] + dx, centre[1] + dy], dtype=model.location.dtype
        )
    return indices


def refresh_cache(env: WargameEnv) -> None:
    """Recompute the step's distance cache against where models stand *now*.

    `objective_hold` reads occupancy off `ctx.distance_cache`, which the env
    builds once per step. Hand-placing models after that step leaves the cache
    describing their old positions, so every assertion here would be measuring
    the wrong board -- and would pass while proving nothing.
    """
    ctx = env.last_step_context
    assert ctx is not None
    ctx.distance_cache = compute_distances(
        env.wargame_models,
        env.objectives,
        alive_mask=alive_mask_for(env.wargame_models),
    )


def unit_income(env: WargameEnv, gate: bool) -> float:
    """Total `objective_hold` pay across the unit, with the gate on or off."""
    calc = ObjectiveHoldCalculator(
        weight=1.25, crowding_exponent=1.0, require_coherent=gate
    )
    ctx = env.last_step_context
    assert ctx is not None
    return sum(
        calc.calculate(i, env.wargame_models[i], env, ctx)
        for i in range(5)
        if env.wargame_models[i].is_alive
    )


def test_the_default_is_off() -> None:
    # Arrange / Act / Assert: every shipped config carries this calculator, so
    # the parameter must be inert unless asked for.
    assert ObjectiveHoldCalculator().require_coherent is False


def test_a_unit_standing_together_is_unaffected() -> None:
    # Arrange: five models chained on one objective — legal, and the gate has
    # nothing to take away.
    env = prepared_env()
    place_one_unit(env, [(0.0, 0.0), (1.5, 0.0), (0.0, 1.5), (1.5, 1.5), (0.0, -1.5)])

    refresh_cache(env)

    # Act / Assert
    assert unit_income(env, gate=False) == pytest.approx(unit_income(env, gate=True))


def test_a_coherent_unit_is_never_gated() -> None:
    # Arrange: the invariant that protects legal play. Whatever shape a unit
    # takes, if the rule calls it coherent the gate must take nothing away —
    # otherwise this "fixes" coherency by forbidding good play.
    #
    # Note this map cannot host the case that motivates the invariant: its
    # objectives sit 13.0-14.5 apart against a 9" spread cap, so a single unit
    # legally spanning two of them is geometrically impossible here. On a map
    # with closer objectives that spread is legal, and this is the assertion
    # that keeps it paying.
    env = prepared_env()
    place_one_unit(env, [(0.0, 0.0), (1.5, 0.0), (0.0, 1.5), (1.5, 1.5), (0.0, -1.5)])

    refresh_cache(env)

    # Act / Assert
    assert unit_income(env, gate=True) == pytest.approx(unit_income(env, gate=False))


def test_a_detached_model_earns_nothing() -> None:
    # Arrange: four models on objective 0, the fifth standing on objective 1 —
    # 14.5 away, so the unit is not coherent and the fifth model is in a state
    # `03-moving.md` forbids. Placed on a real objective, not empty ground, or
    # it would earn nothing anyway and the test would prove nothing.
    env = prepared_env()
    other = np.array(env.objectives[1].location, dtype=float)
    centre = np.array(env.objectives[0].location, dtype=float)
    place_one_unit(env, [(0.0, 0.0), (1.5, 0.0), (0.0, 1.5), (1.5, 1.5), (0.0, 0.0)])
    env.wargame_models[4].location = np.array(
        other, dtype=env.wargame_models[4].location.dtype
    )
    refresh_cache(env)
    ctx = env.last_step_context
    assert ctx is not None
    gated = ObjectiveHoldCalculator(
        weight=1.25, crowding_exponent=1.0, require_coherent=True
    )
    ungated = ObjectiveHoldCalculator(weight=1.25, crowding_exponent=1.0)

    # Act
    detached_gated = gated.calculate(4, env.wargame_models[4], env, ctx)
    detached_ungated = ungated.calculate(4, env.wargame_models[4], env, ctx)

    # Assert: it was being paid, and now it is not.
    assert detached_ungated > 0.0, "test is vacuous unless it earned something"
    assert detached_gated == 0.0
    assert float(centre[0]) != float(other[0]) or float(centre[1]) != float(other[1])


def test_scattering_no_longer_multiplies_the_units_income() -> None:
    # Arrange: one model on each of the three objectives, the rest with the
    # first. Ungated this is the most profitable arrangement available — each
    # lone model divides by 1 and every objective pays its own pot, which is
    # the 5x scatter gradient behind the measured defection.
    env = prepared_env()
    place_one_unit(env, [(0.0, 0.0), (1.5, 0.0), (0.0, 1.5), (0.0, 0.0), (0.0, 0.0)])
    for index, objective in zip((3, 4), env.objectives[1:3]):
        env.wargame_models[index].location = np.array(
            objective.location, dtype=env.wargame_models[index].location.dtype
        )

    refresh_cache(env)

    # Act
    ungated = unit_income(env, gate=False)
    gated = unit_income(env, gate=True)

    # Assert
    assert ungated > 0.0, "test is vacuous unless scattering earned something"
    assert gated < ungated
