"""End-of-Turn attrition: the rules' backstop for a break no move caused.

`03-moving.md` § Regaining coherency removes models one at a time until the unit
is coherent again. It is deliberately the *last* rung: what reaches it is a unit
broken by casualties, not by its own move, which the end-of-move check already
undoes.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.domain.coherency_enforcement import apply_attrition
from wargame_rl.wargame.envs.domain.entities import WargameModel
from wargame_rl.wargame.envs.domain.value_objects import position
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.model.common.factory import create_environment

NEAREST = 2.0
FURTHEST = 9.0


def force(
    locations: list[tuple[float, float]],
    group_ids: list[int] | None = None,
    alive: list[bool] | None = None,
) -> list[WargameModel]:
    """Build a force at fixed positions."""
    group_ids = group_ids if group_ids is not None else [0] * len(locations)
    alive = alive if alive is not None else [True] * len(locations)
    return [
        WargameModel(
            location=position(*loc),
            distances_to_objectives=np.zeros((1, 2)),
            stats={"current_wounds": 1 if is_alive else 0, "max_wounds": 1},
            group_id=group_id,
        )
        for loc, group_id, is_alive in zip(locations, group_ids, alive)
    ]


def survivors(models: list[WargameModel]) -> list[int]:
    """Indices still on the board."""
    return [i for i, m in enumerate(models) if m.is_alive]


def test_a_coherent_force_loses_nobody() -> None:
    # Arrange
    models = force([(0.0, 0.0), (1.5, 0.0), (3.0, 0.0)])

    # Act
    destroyed = apply_attrition(models, NEAREST, FURTHEST)

    # Assert
    assert destroyed == []
    assert survivors(models) == [0, 1, 2]


def test_the_straggler_dies_not_the_body() -> None:
    # Arrange: three together, one detached. Cutting the lone model costs one;
    # cutting the body would cost three, which is why the rule drops from the
    # smallest component first.
    models = force([(0.0, 0.0), (1.5, 0.0), (3.0, 0.0), (30.0, 0.0)])

    # Act
    destroyed = apply_attrition(models, NEAREST, FURTHEST)

    # Assert
    assert destroyed == [3]
    assert survivors(models) == [0, 1, 2]


def test_removal_repeats_until_the_unit_is_whole() -> None:
    # Arrange: a body of two and two separate stragglers. One pass is not
    # enough, and the rule is explicit that models go one at a time.
    models = force([(0.0, 0.0), (1.5, 0.0), (30.0, 0.0), (60.0, 0.0)])

    # Act
    destroyed = apply_attrition(models, NEAREST, FURTHEST)

    # Assert
    assert sorted(destroyed) == [2, 3]
    assert survivors(models) == [0, 1]


def test_a_spread_breach_drops_the_furthest_model() -> None:
    # Arrange: a connected chain that is simply too long -- six models at 2"
    # span 10", past the 9" cap. Connectivity cannot pick a victim here, so the
    # rule falls through to the model furthest from the rest.
    models = force([(float(i) * 2.0, 0.0) for i in range(6)])

    # Act
    destroyed = apply_attrition(models, NEAREST, FURTHEST)

    # Assert: an end model, not a middle one.
    assert destroyed
    assert destroyed[0] in (0, 5)
    assert survivors(models)


def test_attrition_is_self_limiting() -> None:
    # Arrange: a unit scattered to the four corners. It must not eat itself
    # entirely -- a unit of one model is coherent by definition, so the removal
    # loop always terminates with a survivor.
    models = force([(0.0, 0.0), (50.0, 0.0), (0.0, 40.0), (50.0, 40.0)])

    # Act
    apply_attrition(models, NEAREST, FURTHEST)

    # Assert
    assert len(survivors(models)) >= 1


def test_units_are_regained_independently() -> None:
    # Arrange: unit 0 is broken, unit 1 is tight.
    models = force(
        [(0.0, 0.0), (30.0, 0.0), (10.0, 20.0), (11.5, 20.0)],
        group_ids=[0, 0, 1, 1],
    )

    # Act
    destroyed = apply_attrition(models, NEAREST, FURTHEST)

    # Assert: unit 1 is untouched.
    assert 2 not in destroyed and 3 not in destroyed
    assert len(destroyed) == 1


def test_already_dead_models_do_not_drag_a_unit_down() -> None:
    # Arrange: two survivors together, a casualty lying far off. The dead are
    # not in the unit for this test, so nothing further should die.
    models = force([(0.0, 0.0), (1.5, 0.0), (60.0, 40.0)], alive=[True, True, False])

    # Act
    destroyed = apply_attrition(models, NEAREST, FURTHEST)

    # Assert
    assert destroyed == []
    assert survivors(models) == [0, 1]


def test_the_removal_choice_is_deterministic() -> None:
    # Arrange: the same broken force twice. A seeded env and the bit-identical
    # golden gates both require this.
    layout = [(0.0, 0.0), (1.5, 0.0), (30.0, 0.0), (60.0, 0.0)]

    # Act
    first = apply_attrition(force(layout), NEAREST, FURTHEST)
    second = apply_attrition(force(layout), NEAREST, FURTHEST)

    # Assert
    assert first == second


def test_attrition_is_off_by_default_and_on_when_asked() -> None:
    # Arrange
    with open("configs/golden/25v25_shooting_opponent.yaml") as handle:
        config = cast(
            WargameEnvConfig, parse_yaml_raw_as(WargameEnvConfig, handle.read())
        )
    config.render_mode = None

    # Act / Assert
    assert not config.coherency.attrition
    env = create_environment(env_config=config)
    env.reset(seed=700000)
    assert env is not None


@pytest.mark.parametrize("n_models", [2, 3, 5])
def test_a_unit_never_loses_every_model(n_models: int) -> None:
    # Arrange: maximally scattered, whatever the size.
    models = force([(float(i) * 25.0, 0.0) for i in range(n_models)])

    # Act
    destroyed = apply_attrition(models, NEAREST, FURTHEST)

    # Assert
    assert len(destroyed) == n_models - 1
    assert len(survivors(models)) == 1
