"""The eval-time coherency metrics: `coherency_rate` and `models_out_of_coherency`.

Coherency was measurable only offline, against a finished checkpoint, which is
the wrong place for it while a run trains *under* the rule -- a run can drift out
of formation for a thousand epochs with nothing in the dashboard to say so.

The load-bearing test is `test_coherent_deployment_reads_higher_than_default`:
a metric that reported the same number whichever way the army was placed would
pass every structural assertion and still be worthless.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.env_components.coherency_tracker import CoherencyTracker
from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv

NEAREST = 2.0
FURTHEST = 9.0
GOLDEN = "configs/golden/25v25_shooting_opponent.yaml"


def make_config(coherent_deployment: bool) -> WargameEnvConfig:
    """The golden scenario, deployed either by the rule or by default."""
    with open(GOLDEN) as handle:
        config = cast(
            WargameEnvConfig, parse_yaml_raw_as(WargameEnvConfig, handle.read())
        )
    config.render_mode = None
    config.coherency.enforce_at_deployment = coherent_deployment
    return config


def record_once(
    tracker: CoherencyTracker,
    positions: list[tuple[float, float]],
    group_ids: list[int],
    alive: list[bool] | None = None,
) -> None:
    """Fold one hand-placed formation into a tracker."""
    n = len(positions)
    tracker.record(
        positions=np.array(positions, dtype=float),
        group_ids=np.array(group_ids, dtype=np.intp),
        alive_mask=np.array(alive if alive is not None else [True] * n, dtype=bool),
        base_radii=np.zeros(n, dtype=float),
        nearest_distance=NEAREST,
        furthest_distance=FURTHEST,
    )


def test_nothing_sampled_reports_none_rather_than_zero() -> None:
    # Arrange / Act: a fresh tracker has seen no phases.
    tracker = CoherencyTracker()

    # Assert: None, not 0.0 -- a zero would plot as "never in coherency".
    assert tracker.coherency_rate is None
    assert tracker.models_out_of_coherency is None


def test_a_coherent_formation_scores_one_and_nobody_out() -> None:
    # Arrange: one chained unit well inside both caps.
    tracker = CoherencyTracker()

    # Act
    record_once(tracker, [(0.0, 0.0), (1.5, 0.0), (3.0, 0.0)], [0, 0, 0])

    # Assert
    assert tracker.coherency_rate == 1.0
    assert tracker.models_out_of_coherency == 0.0


def test_a_detached_model_is_counted_out() -> None:
    # Arrange: two chained, one stranded 20 away.
    tracker = CoherencyTracker()

    # Act
    record_once(tracker, [(0.0, 0.0), (1.5, 0.0), (20.0, 0.0)], [0, 0, 0])

    # Assert: the unit is broken, and exactly one model is off the body.
    assert tracker.coherency_rate == 0.0
    assert tracker.models_out_of_coherency == 1.0


def test_the_rate_averages_over_units_and_the_count_over_phases() -> None:
    # Arrange: one coherent unit and one broken one, in the same phase.
    tracker = CoherencyTracker()

    # Act
    record_once(
        tracker,
        [(0.0, 0.0), (1.5, 0.0), (40.0, 0.0), (41.5, 0.0), (41.5, 20.0)],
        [0, 0, 1, 1, 1],
    )

    # Assert: 1 of 2 units coherent, and the one stranded model counted once.
    assert tracker.coherency_rate == 0.5
    assert tracker.models_out_of_coherency == 1.0


def test_the_dead_are_not_counted_out() -> None:
    # Arrange: the same stranded geometry, but the straggler is a casualty.
    # Regression: a corpse left at the far end of the board would otherwise read
    # as a permanent coherency breach for the rest of the episode.
    tracker = CoherencyTracker()

    # Act
    record_once(
        tracker,
        [(0.0, 0.0), (1.5, 0.0), (20.0, 0.0)],
        [0, 0, 0],
        alive=[True, True, False],
    )

    # Assert
    assert tracker.coherency_rate == 1.0
    assert tracker.models_out_of_coherency == 0.0


def test_the_metric_is_none_before_a_movement_phase_and_set_after() -> None:
    # Arrange
    env = WargameEnv(make_config(coherent_deployment=True))
    env.reset(seed=7)

    # Act: step until the env has resolved at least one movement phase.
    for _ in range(4):
        env.step(WargameEnvAction(actions=list(env.action_space.sample())))
        if env.coherency_rate is not None:
            break

    # Assert
    assert env.coherency_rate is not None
    assert 0.0 <= env.coherency_rate <= 1.0
    assert env.models_out_of_coherency is not None


@pytest.mark.parametrize("seed", [3, 11, 29])
def test_coherent_deployment_reads_higher_than_default(seed: int) -> None:
    # Arrange: the same scenario placed both ways. This is the control that
    # makes the metric worth logging -- one that could not separate these two
    # would satisfy every other test here and still measure nothing.
    readings = {}
    for coherent in (False, True):
        env = WargameEnv(make_config(coherent_deployment=coherent))
        env.reset(seed=seed)
        # Everyone holds, so the reading is the deployment and nothing else.
        env.step(WargameEnvAction(actions=[0] * len(env.wargame_models)))
        readings[coherent] = (env.coherency_rate, env.models_out_of_coherency)

    # Assert: the separation is total, not merely non-negative. Default
    # placement anchors each model on one *random* squadmate, which bounds the
    # nearest neighbour and leaves the unit's span unbounded, so it deploys
    # coherent essentially never; the rule deploys coherent by construction.
    # A `>=` here would pass on a metric that returned a constant.
    assert readings[True] == (1.0, 0.0)
    coherent_rate, models_out = readings[False]
    assert coherent_rate == 0.0
    assert models_out is not None and models_out > 5
