"""The coherency inputs, and the corpse bug in the cohesion column.

`observe_coherency` adds the two halves of the rule the observation never
carried. The corpse fix is separate and unconditional: the same-group column had
no alive filter, so it could report the distance to a body.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.domain.entities import WargameModel
from wargame_rl.wargame.envs.domain.value_objects import position
from wargame_rl.wargame.envs.env_components.observation_builder import (
    CoherencyDistances,
    _coherency_features,
    _models_to_obs,
    build_observation,
)
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.model.common.factory import create_environment
from wargame_rl.wargame.model.common.observation import (
    _same_group_closest_distance,
    observation_to_tensor,
)
from wargame_rl.wargame.model.net import TransformerNetwork

RULES = CoherencyDistances(nearest=2.0, furthest=9.0)


def make_models(
    locations: list[tuple[float, float]],
    group_ids: list[int],
    alive: list[bool] | None = None,
    base_radius: float = 0.0,
) -> list[WargameModel]:
    """Build a force at fixed positions, some of it optionally dead."""
    alive = alive if alive is not None else [True] * len(locations)
    return [
        WargameModel(
            location=position(*loc),
            distances_to_objectives=np.zeros((1, 2)),
            stats={"current_wounds": 1 if is_alive else 0, "max_wounds": 1},
            group_id=group_id,
            base_radius=base_radius,
        )
        for loc, group_id, is_alive in zip(locations, group_ids, alive)
    ]


def test_the_cohesion_column_ignores_the_dead() -> None:
    # Arrange: a model with a corpse at its feet and a live squadmate 20 away.
    # The corpse keeps its position forever -- `take_damage` writes only wounds.
    locations = [(0.0, 0.0), (1.0, 0.0), (20.0, 0.0)]
    locs = np.array(locations, dtype=np.float32)
    group_ids = np.array([0, 0, 0], dtype=np.int32)

    # Act
    with_corpse = _same_group_closest_distance(
        locs, group_ids, 100.0, np.array([True, False, True])
    )
    all_alive = _same_group_closest_distance(
        locs, group_ids, 100.0, np.array([True, True, True])
    )

    # Assert: the living squadmate at 20, not the body at 1.
    assert with_corpse[0, 0] == np.float32(0.2)
    assert all_alive[0, 0] == np.float32(0.01)


def test_a_model_whose_whole_unit_is_dead_reads_as_alone() -> None:
    # Arrange: the sole survivor of a wiped squad. Its dead mates must not
    # register as company.
    locs = np.array([(0.0, 0.0), (1.0, 0.0)], dtype=np.float32)
    group_ids = np.array([0, 0], dtype=np.int32)

    # Act
    column = _same_group_closest_distance(
        locs, group_ids, 100.0, np.array([True, False])
    )

    # Assert: 1.0 is this column's "no same-group model" value.
    assert column[0, 0] == np.float32(1.0)


def test_the_spread_column_sees_what_the_nearest_column_cannot() -> None:
    # Arrange: five models chained at 2", spanning 8". Every model has a
    # neighbour at 2, so the nearest-neighbour column is identical for the
    # middle of the unit and cannot express the span at all.
    models = make_models([(float(i) * 2.0, 0.0) for i in range(5)], [0] * 5)

    # Act
    spread, component = _coherency_features(models, RULES.nearest, RULES.furthest)

    # Assert: the end models are 8 of 9 apart, the middle ones less.
    assert spread[0] == np.float32(8.0 / 9.0)
    assert spread[2] < spread[0]
    assert (component == 1.0).all()


def test_the_component_column_reports_a_split_unit() -> None:
    # Arrange: a body of three and a detached pair, past the 2" chain.
    models = make_models(
        [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (7.0, 0.0), (8.0, 0.0)], [0] * 5
    )

    # Act
    _spread, component = _coherency_features(models, RULES.nearest, RULES.furthest)

    # Assert: three fifths and two fifths, so every model can tell which side of
    # the break it is on and how much of its unit went with it.
    np.testing.assert_allclose(component, [0.6, 0.6, 0.6, 0.4, 0.4], rtol=1e-6)


def test_a_coherent_unit_reports_one_whole_component() -> None:
    # Arrange
    models = make_models([(0.0, 0.0), (1.5, 0.0), (3.0, 0.0)], [0] * 3)

    # Act
    _spread, component = _coherency_features(models, RULES.nearest, RULES.furthest)

    # Assert
    assert (component == 1.0).all()


def test_the_spread_column_saturates_rather_than_running_away() -> None:
    # Arrange: a pair 90" apart, ten times the cap.
    models = make_models([(0.0, 0.0), (90.0, 0.0)], [0, 0])

    # Act
    spread, _component = _coherency_features(models, RULES.nearest, RULES.furthest)

    # Assert: clipped at 1, so one stranded model cannot dominate the column.
    assert spread.tolist() == [1.0, 1.0]


def test_dead_models_report_compliant_values() -> None:
    # Arrange: a casualty far from its unit. It must not read as a violation --
    # the row is masked by `alive` anyway, and a violation there would be a
    # standing penalty signal on every corpse.
    models = make_models(
        [(0.0, 0.0), (1.5, 0.0), (80.0, 40.0)], [0] * 3, alive=[True, True, False]
    )

    # Act
    spread, component = _coherency_features(models, RULES.nearest, RULES.furthest)

    # Assert
    assert spread[2] == 0.0
    assert component[2] == 1.0
    # And the survivors are judged as a unit of two, not of three.
    assert spread[0] == np.float32(1.5 / 9.0)


def test_the_columns_are_absent_unless_asked_for() -> None:
    # Arrange
    models = make_models([(0.0, 0.0), (1.0, 0.0)], [0, 0])

    # Act
    without = _models_to_obs(models, max_groups=2)
    with_coherency = _models_to_obs(models, max_groups=2, coherency=RULES)

    # Assert: None keeps the token at its historical width.
    assert without[0].coherency_spread is None
    assert without[0].coherency_component is None
    assert with_coherency[0].coherency_spread is not None
    assert with_coherency[0].size == without[0].size + 2


def test_the_alive_column_is_still_where_the_mask_looks() -> None:
    # Arrange: the trap these two columns had to dodge. `_alive_feature_index`
    # counts *backwards* from the last column, so a column appended after the
    # combat stats would make the key-padding mask read `wound_ratio` as
    # `alive` -- dead models stay attendable, live ones drop out, and nothing
    # raises. Anchored to env truth rather than to a recorded index, and with
    # one model genuinely dead so an off-by-one is detectable at all.
    with open("configs/golden/25v25_shooting_opponent.yaml") as handle:
        config = cast(
            WargameEnvConfig, parse_yaml_raw_as(WargameEnvConfig, handle.read())
        )
    config.render_mode = None
    config.observe_coherency = True
    env = create_environment(env_config=config)
    env.reset(seed=700000)
    # Build the network first: `policy_from_env` resets the env, so killing a
    # model before this would be undone.
    net = TransformerNetwork.policy_from_env(env)
    env.wargame_models[3].take_damage(1)

    # Act
    tensors = observation_to_tensor(build_observation(env))
    feature_dim = tensors[2].shape[-1]
    index = net._alive_feature_index(feature_dim, len(env.opponent_models))

    # Assert
    column = tensors[2][:, index].cpu().numpy()
    truth = np.array(
        [1.0 if m.is_alive else 0.0 for m in env.wargame_models], dtype=np.float32
    )
    assert not truth.all(), "the dead model was revived; the test proves nothing"
    np.testing.assert_array_equal(column, truth)


def test_distances_are_base_to_base() -> None:
    # Arrange: the same pair, as point models and with 32mm bases.
    locations = [(0.0, 0.0), (5.0, 0.0)]

    # Act
    points, _c = _coherency_features(
        make_models(locations, [0, 0]), RULES.nearest, RULES.furthest
    )
    based, _c2 = _coherency_features(
        make_models(locations, [0, 0], base_radius=0.63),
        RULES.nearest,
        RULES.furthest,
    )

    # Assert: the bases shorten the gap by their diameter.
    assert points[0] == np.float32(5.0 / 9.0)
    assert based[0] < points[0]
