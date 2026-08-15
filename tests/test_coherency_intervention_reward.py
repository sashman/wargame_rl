"""`coherency_intervention`: charge a model for a move the referee corrected.

This term exists because end-of-move enforcement supplies **no gradient at all**
inside the set of moves it reverts -- every one of them produces the identical
outcome, so they share an advantage and the policy gradient there is exactly
zero. Measured consequence: a policy trained under enforcement eroded from 0.847
intended coherency to 0.630, worse than never training under the rule.

The load-bearing test is `test_it_is_zero_when_enforcement_is_off`. A term that
charged models regardless of whether the referee acted would be a flat movement
tax wearing this term's name, would satisfy every other assertion here, and
would suppress movement exactly as `revert_unit` does.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.reward.calculators.coherency_intervention import (
    CoherencyInterventionCalculator,
)
from wargame_rl.wargame.envs.reward.calculators.registry import CALCULATOR_REGISTRY
from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv

GOLDEN = "configs/golden/25v25_shooting_opponent.yaml"


def build(enforce_move: str) -> WargameEnv:
    """The golden scenario with coherent deployment and the given enforcement."""
    with open(GOLDEN) as handle:
        config = cast(
            WargameEnvConfig, parse_yaml_raw_as(WargameEnvConfig, handle.read())
        )
    config.render_mode = None
    config.coherency.enforce_at_deployment = True
    config.coherency.enforce_move = enforce_move
    return WargameEnv(config)


def total_charge(env: WargameEnv, penalty: float = -0.02) -> float:
    """This term's total across the force for the step just taken."""
    calculator = CoherencyInterventionCalculator(penalty=penalty)
    ctx = env.last_step_context
    assert ctx is not None
    return sum(
        calculator.calculate(i, model, env, ctx)
        for i, model in enumerate(env.wargame_models)
    )


def step_until_movement(env: WargameEnv, steps: int = 4) -> None:
    """Take random actions, which a coherent force cannot keep legal."""
    for _ in range(steps):
        env.step(WargameEnvAction(actions=list(env.action_space.sample())))


def test_it_is_registered_under_its_config_name() -> None:
    # Arrange / Act / Assert: unreachable from YAML otherwise.
    assert CALCULATOR_REGISTRY["coherency_intervention"] is (
        CoherencyInterventionCalculator
    )


@pytest.mark.parametrize("penalty", [0.01, 1.0])
def test_a_positive_penalty_is_rejected(penalty: float) -> None:
    # Arrange / Act / Assert: a positive value would PAY for illegal moves,
    # which is precisely the defect this term exists to remove.
    with pytest.raises(ValueError, match="penalty must be <= 0"):
        CoherencyInterventionCalculator(penalty=penalty)


def test_it_is_zero_when_enforcement_is_off() -> None:
    # Arrange: no referee, so nothing is ever corrected and the term must be
    # silent. Without this, the term could be a flat movement tax and every
    # other test here would still pass.
    env = build("off")
    env.reset(seed=5)

    # Act
    step_until_movement(env)

    # Assert
    assert total_charge(env) == 0.0


def test_it_charges_when_the_referee_intervenes() -> None:
    # Arrange: the same scenario and seed with enforcement on. Random moves
    # break a coherent force immediately, so the referee has work to do.
    env = build("revert_model")
    env.reset(seed=5)

    # Act
    step_until_movement(env)

    # Assert: something was corrected, and the term charged for exactly that.
    displaced = env.last_step_context.models_displaced_by_enforcement  # type: ignore[union-attr]
    assert displaced is not None
    assert displaced.any(), "test is vacuous unless enforcement actually fired"
    assert total_charge(env) == pytest.approx(-0.02 * int(displaced.sum()))


def test_only_the_corrected_models_are_charged() -> None:
    # Arrange: the charge must be per-model differentiated, not spread over the
    # force -- a flat charge gives no model a private reason to move legally,
    # which is the mistake flat `objective_hold` made.
    env = build("revert_model")
    env.reset(seed=11)

    # Act
    step_until_movement(env)
    ctx = env.last_step_context
    assert ctx is not None
    displaced = ctx.models_displaced_by_enforcement
    assert displaced is not None and displaced.any()
    calculator = CoherencyInterventionCalculator(penalty=-0.05)
    charges = np.array(
        [
            calculator.calculate(i, model, env, ctx)
            for i, model in enumerate(env.wargame_models)
        ]
    )

    # Assert: charged exactly where the referee acted, nowhere else.
    alive = np.array([m.is_alive for m in env.wargame_models], dtype=bool)
    assert np.array_equal(charges < 0.0, displaced & alive)


def test_the_dead_are_not_charged() -> None:
    # Arrange: a casualty cannot choose a legal move, so charging it would be a
    # standing penalty for the rest of the episode -- the same defect the
    # coherency observation had with corpses.
    env = build("revert_model")
    env.reset(seed=5)
    step_until_movement(env)
    ctx = env.last_step_context
    assert ctx is not None
    displaced = ctx.models_displaced_by_enforcement
    assert displaced is not None and displaced.any()
    index = int(np.flatnonzero(displaced)[0])
    env.wargame_models[index].stats["current_wounds"] = 0

    # Act
    calculator = CoherencyInterventionCalculator()
    charge = calculator.calculate(index, env.wargame_models[index], env, ctx)

    # Assert
    assert charge == 0.0
