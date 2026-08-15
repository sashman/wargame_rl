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

    # Assert: the unit is broken, and every model is out of coherency.
    #
    # Three, not one. The SPREAD condition is collective -- once the straggler
    # is beyond the 9" cap, no model in the unit is within the cap of every
    # other, so none of them satisfies the rule. This counts models the RULE
    # calls out (`member_coherency`), which is what `just measure-coherency`
    # has always reported. The tracker used to count `size -
    # largest_component_size`, i.e. the chain graph only, which gave 1 here and
    # **0 for any spread-only breach** -- a unit 100% in breach reporting no
    # models adrift. The two metrics of the same name disagreed; they no longer
    # do.
    assert tracker.coherency_rate == 0.0
    assert tracker.models_out_of_coherency == 3.0


def test_the_rate_averages_over_units_and_the_count_over_phases() -> None:
    # Arrange: one coherent unit and one broken one, in the same phase.
    tracker = CoherencyTracker()

    # Act
    record_once(
        tracker,
        [(0.0, 0.0), (1.5, 0.0), (40.0, 0.0), (41.5, 0.0), (41.5, 20.0)],
        [0, 0, 1, 1, 1],
    )

    # Assert: 1 of 2 units coherent. The broken unit contributes all three of
    # its models, since the 20-unit straggler puts every member outside the 9"
    # spread cap -- see `test_a_detached_model_is_counted_out`.
    assert tracker.coherency_rate == 0.5
    assert tracker.models_out_of_coherency == 3.0


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


def test_a_spread_only_breach_is_counted() -> None:
    # Arrange: six models in a line at 2.0 spacing. The chain is intact, so the
    # unit is ONE connected component and a `size - largest_component_size`
    # count sees nothing wrong -- but the line spans 10.0 against the 9" cap,
    # so the unit is entirely in breach.
    #
    # Regression: this is the case the tracker structurally could not see, and
    # no test in this file constructed it -- every other one splits the chain
    # graph. `just measure-coherency` reported it correctly throughout, so the
    # training metric silently undercounted exactly the breach category that
    # turned out to dominate.
    tracker = CoherencyTracker()

    # Act
    record_once(tracker, [(float(i) * 2.0, 0.0) for i in range(6)], [0] * 6)

    # Assert
    assert tracker.coherency_rate == 0.0
    adrift = tracker.models_out_of_coherency
    assert adrift is not None and adrift > 0.0


def test_intent_and_board_diverge_under_enforcement() -> None:
    """The metric pair that would have caught this project's worst error.

    `coherency_rate` is sampled AFTER the end-of-move revert, so under
    `enforce_move` it is legal by construction and says nothing about the
    policy. A whole investigation published "1.000 compliance" for weights that
    intend 0.630 on exactly this confusion.

    So: same config, same weights, same layouts, enforcement on. The board must
    read (near) perfect while intent reads materially worse -- and if they ever
    agree, the intent metric has stopped measuring what it claims.
    """
    config = make_config(coherent_deployment=True)
    config.coherency.enforce_move = "revert_model"
    env = WargameEnv(config)
    env.reset(seed=3)

    # Act: random-ish movement, which a coherent force cannot survive legally.
    for _ in range(6):
        env.step(WargameEnvAction(actions=list(env.action_space.sample())))

    board = env.coherency_rate
    intent = env.intended_coherency_rate
    assert board is not None and intent is not None

    # Assert: the referee delivers a legal board, the policy did not choose one.
    assert board > intent, (
        f"intent {intent:.3f} should be worse than the enforced board {board:.3f}; "
        "if they match, the intent metric is being sampled after enforcement"
    )
    assert env.models_reverted_last_move >= 0
