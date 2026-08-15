"""`BaselineResult` carries its per-episode values, so a mean can have an SE.

The loop always built these lists and then threw them away, which is why no
figure in any report here has ever carried an error bar -- and why a string of
noise-level gaps was read as a sequence of effects. Per-episode `vp_margin` has
a standard deviation of 45-50 on the 25v25 scenarios, so at n=30 the SE is ~8-9,
larger than most arm differences ever measured.

The load-bearing test is `test_the_per_episode_margins_are_the_episodes_own`.
Arrays that merely had the right *length* would satisfy every other assertion
here while carrying the wrong episodes, and pairing two results on them would
then produce a confident number about nothing.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from pydantic_yaml import parse_yaml_raw_as

from wargame_rl.wargame.envs.baseline.evaluate import (
    BaselineResult,
    evaluate_baseline,
    paired_difference,
    standard_error,
)
from wargame_rl.wargame.envs.baseline.registry import build_baseline_policy
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.wargame import WargameEnv

CONFIG = "configs/golden/25v25_shooting_opponent.yaml"
SEEDS = [700000, 700001, 700002]


def build_env() -> WargameEnv:
    """The golden shooting scenario, headless."""
    with open(CONFIG) as handle:
        config = cast(
            WargameEnvConfig, parse_yaml_raw_as(WargameEnvConfig, handle.read())
        )
    config.render_mode = None
    return WargameEnv(config)


def score(name: str) -> BaselineResult:
    """Run a scripted baseline over the fixed seed set."""
    return evaluate_baseline(build_baseline_policy(name), build_env(), SEEDS)


def test_standard_error_needs_two_samples() -> None:
    # Arrange / Act / Assert: one episode has no spread to report, and 0.0
    # would read as "measured, and exactly certain".
    assert standard_error([]) is None
    assert standard_error([12.0]) is None
    assert standard_error([0.0, 2.0]) == pytest.approx(1.0)


def test_the_per_episode_margins_are_the_episodes_own() -> None:
    # Arrange: arrays of the right length but the wrong contents would pass
    # every other test here and make a paired comparison meaningless.
    result = score("squad_march_shoot")

    # Act / Assert: each entry is that episode's own margin, and they average
    # to the reported mean.
    assert len(result.vp_margin_per_episode) == len(SEEDS)
    assert float(np.mean(result.vp_margin_per_episode)) == pytest.approx(
        result.vp_margin
    )
    assert float(np.mean(result.objectives_held_per_episode)) == pytest.approx(
        result.objectives_held
    )
    assert float(np.mean(result.win_per_episode)) == pytest.approx(result.win_rate)


def test_the_arrays_are_in_seed_order() -> None:
    # Arrange: pairing is only valid if entry i of both results is the same
    # layout, which requires seed order and not, say, sorted values.
    full = score("squad_march_shoot")
    first_seed_only = evaluate_baseline(
        build_baseline_policy("squad_march_shoot"), build_env(), SEEDS[:1]
    )

    # Act / Assert: scoring seed 0 alone reproduces entry 0 of the full run.
    assert first_seed_only.vp_margin_per_episode[0] == pytest.approx(
        full.vp_margin_per_episode[0]
    )


def test_a_paired_difference_is_the_mean_of_the_per_episode_gaps() -> None:
    # Arrange: two policies on identical layouts, which is the only case where
    # differencing cancels the layout variance that dominates this scenario.
    treatment = score("squad_march_shoot")
    control = score("squad_march")

    # Act
    difference, standard_err = paired_difference(treatment, control)

    # Assert
    assert difference == pytest.approx(treatment.vp_margin - control.vp_margin)
    assert standard_err is not None


def test_pairing_across_different_episode_counts_is_refused() -> None:
    # Arrange: silently truncating would difference episode i of one layout set
    # against episode i of another, which is not a paired comparison at all.
    treatment = score("squad_march_shoot")
    control = evaluate_baseline(
        build_baseline_policy("squad_march"), build_env(), SEEDS[:2]
    )

    # Act / Assert
    with pytest.raises(ValueError, match="equal, non-empty episode counts"):
        paired_difference(treatment, control)
