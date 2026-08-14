"""The `unit_coherency` reward: pay a model for standing with its squad.

Every other coherency mechanism here is a constraint. This is the only one that
gives a policy a *reason* to hold formation, so the tests it needs are about the
gradient it creates, not about whether it returns a number.

The load-bearing case is `test_the_straggler_and_its_squadmates_are_paid_apart`:
a term paying every model in a broken unit the same amount would satisfy every
other assertion here and create no gradient at all -- the mistake flat
`objective_hold` made, where the thirteenth model on a point earned what the
first did and no model ever had a private reason to leave.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from pydantic import ValidationError
from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.reward.calculators.registry import CALCULATOR_REGISTRY
from wargame_rl.wargame.envs.reward.calculators.unit_coherency import (
    UnitCoherencyCalculator,
)
from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv

GOLDEN = "configs/golden/25v25_shooting_opponent.yaml"


def load_config() -> WargameEnvConfig:
    """The golden scenario, parsed off disk."""
    with open(GOLDEN) as handle:
        config = cast(
            WargameEnvConfig, parse_yaml_raw_as(WargameEnvConfig, handle.read())
        )
    config.render_mode = None
    return config


def prepared_env() -> WargameEnv:
    """A reset env that has taken one step, so a `StepContext` exists.

    The context is only a cache key for this calculator -- it reads positions
    off the models directly -- but `calculate` takes one, and `reset` alone
    leaves `last_step_context` unset.
    """
    env = WargameEnv(load_config())
    env.reset(seed=5)
    env.step(WargameEnvAction(actions=[0] * len(env.wargame_models)))
    return env


def place(env: WargameEnv, positions: list[tuple[float, float]]) -> None:
    """Put the first len(positions) player models at hand-chosen points."""
    for model, (x, y) in zip(env.wargame_models, positions):
        model.location = np.array([x, y], dtype=model.location.dtype)


def payments(env: WargameEnv, calculator: UnitCoherencyCalculator) -> list[float]:
    """This calculator's value for every player model, in index order."""
    ctx = env.last_step_context
    assert ctx is not None
    return [
        calculator.calculate(i, model, env, ctx)
        for i, model in enumerate(env.wargame_models)
    ]


def test_it_is_registered_under_its_config_name() -> None:
    # Arrange / Act / Assert: a calculator absent from the registry cannot be
    # named in YAML, which is the only way a run would ever use it.
    assert CALCULATOR_REGISTRY["unit_coherency"] is UnitCoherencyCalculator


@pytest.mark.parametrize("value", [-0.01, -1.0])
def test_a_negative_value_is_rejected(value: float) -> None:
    # Arrange / Act / Assert: the term pays for coherency; a negative `value`
    # would silently invert it into a fine for staying together.
    with pytest.raises(ValueError, match="value must be >= 0"):
        UnitCoherencyCalculator(value=value)


def test_a_positive_straggler_penalty_is_rejected() -> None:
    # Arrange / Act / Assert: a positive "penalty" would pay stragglers.
    with pytest.raises(ValueError, match="straggler_penalty must be <= 0"):
        UnitCoherencyCalculator(straggler_penalty=0.5)


def test_a_coherent_unit_is_paid_in_full() -> None:
    # Arrange: one chained unit, everyone inside both caps.
    env = prepared_env()
    calculator = UnitCoherencyCalculator(value=0.05)
    for index, model in enumerate(env.wargame_models):
        model.group_id = 0
        model.stats["current_wounds"] = 1 if index < 3 else 0
    place(env, [(10.0, 10.0), (11.5, 10.0), (13.0, 10.0)])

    # Act
    paid = payments(env, calculator)

    # Assert
    assert paid[:3] == [0.05, 0.05, 0.05]


def test_the_straggler_and_its_squadmates_are_paid_apart() -> None:
    # Arrange: two models chained, the third stranded far enough to break the
    # chain *and* the spread. This is the whole point of the term -- if these
    # three are paid the same, no model has a private reason to close up.
    env = prepared_env()
    calculator = UnitCoherencyCalculator(value=0.05)
    for index, model in enumerate(env.wargame_models):
        model.group_id = 0
        model.stats["current_wounds"] = 1 if index < 3 else 0
    place(env, [(10.0, 10.0), (11.5, 10.0), (35.0, 10.0)])

    # Act
    paid = payments(env, calculator)

    # Assert: the body earns, the straggler does not.
    assert paid[0] == 0.05
    assert paid[1] == 0.05
    assert paid[2] == 0.0
    assert paid[0] != paid[2], "a flat payment creates no gradient"


def test_the_straggler_penalty_sharpens_the_same_gap() -> None:
    # Arrange: the same geometry, with the optional extra charge set.
    env = prepared_env()
    calculator = UnitCoherencyCalculator(value=0.05, straggler_penalty=-0.02)
    for index, model in enumerate(env.wargame_models):
        model.group_id = 0
        model.stats["current_wounds"] = 1 if index < 3 else 0
    place(env, [(10.0, 10.0), (11.5, 10.0), (35.0, 10.0)])

    # Act
    paid = payments(env, calculator)

    # Assert
    assert paid[2] == -0.02
    assert paid[0] - paid[2] == pytest.approx(0.07)


def test_a_lone_survivor_is_paid_rather_than_fined() -> None:
    # Arrange: one model left in its unit. The rule makes a one-model unit
    # coherent by definition, and fining it would price a casualty this model
    # did not cause -- the same defect the corpse bug had in the observation.
    env = prepared_env()
    calculator = UnitCoherencyCalculator(value=0.05, straggler_penalty=-0.02)
    for index, model in enumerate(env.wargame_models):
        model.group_id = 0
        model.stats["current_wounds"] = 1 if index == 0 else 0
    place(env, [(10.0, 10.0)])

    # Act
    paid = payments(env, calculator)

    # Assert
    assert paid[0] == 0.05


def test_the_dead_are_paid_nothing() -> None:
    # Arrange: a corpse sitting inside a coherent body.
    env = prepared_env()
    calculator = UnitCoherencyCalculator(value=0.05)
    for index, model in enumerate(env.wargame_models):
        model.group_id = 0
        model.stats["current_wounds"] = 1 if index < 3 else 0
    env.wargame_models[1].stats["current_wounds"] = 0
    place(env, [(10.0, 10.0), (11.5, 10.0), (13.0, 10.0)])

    # Act
    paid = payments(env, calculator)

    # Assert
    assert paid[1] == 0.0


def test_units_are_scored_independently() -> None:
    # Arrange: one intact unit and one broken one, far apart. A term that
    # evaluated the army as a whole would fail every model in both.
    env = prepared_env()
    calculator = UnitCoherencyCalculator(value=0.05)
    for index, model in enumerate(env.wargame_models):
        model.stats["current_wounds"] = 1 if index < 5 else 0
        model.group_id = 0 if index < 2 else 1
    place(
        env,
        [
            (10.0, 10.0),
            (11.5, 10.0),
            (40.0, 30.0),
            (41.5, 30.0),
            (41.5, 5.0),
        ],
    )

    # Act
    paid = payments(env, calculator)

    # Assert: unit 0 whole and paid; unit 1 broken, body paid, straggler not.
    assert paid[:2] == [0.05, 0.05]
    assert paid[2] == 0.05
    assert paid[3] == 0.05
    assert paid[4] == 0.0


def test_a_config_using_the_reward_without_the_observation_is_rejected() -> None:
    # Arrange: the desk check made mechanical. Without `observe_coherency` the
    # two states this term separates are identical in the observation, and an
    # unattributable reward is experienced only as "this pays less" -- the
    # failure mode that cost GPU hours on the overstack penalty and on
    # `objective_hold.surplus_value`.
    config = load_config()
    config.observe_coherency = False
    payload = config.model_dump()
    payload["reward_phases"][0]["reward_calculators"].append(
        {"type": "unit_coherency", "weight": 1.0, "params": {"value": 0.05}}
    )

    # Act / Assert
    with pytest.raises(ValidationError, match="observe_coherency"):
        WargameEnvConfig(**payload)


def test_the_same_config_is_accepted_once_the_observation_is_on() -> None:
    # Arrange: the control for the test above. A guard that rejected the pair
    # unconditionally would pass that test and make the calculator unusable.
    config = load_config()
    config.observe_coherency = True
    payload = config.model_dump()
    payload["reward_phases"][0]["reward_calculators"].append(
        {"type": "unit_coherency", "weight": 1.0, "params": {"value": 0.05}}
    )

    # Act
    accepted = WargameEnvConfig(**payload)

    # Assert
    assert accepted.observe_coherency is True
