"""`all_objectives_occupied`: the success criterion that counts POINTS, not
models -- every objective has at least one alive player model inside it.
The two on file (`all_at_objectives`, `fraction_at_objectives`) cannot tell
twelve models on one point from three on each of four, which is the
question the curriculum's spread rung asks (#340 A3)."""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.envs.env_components.distance_cache import compute_distances
from wargame_rl.wargame.envs.reward.criteria.all_objectives_occupied import (
    AllObjectivesOccupiedCriteria,
)
from wargame_rl.wargame.envs.reward.criteria.registry import (
    CRITERIA_REGISTRY,
    build_criteria,
)
from wargame_rl.wargame.envs.reward.phase import (
    RewardCalculatorConfig,
    RewardPhaseConfig,
    SuccessCriteriaConfig,
)
from wargame_rl.wargame.envs.reward.step_context import StepContext
from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.types.config.entities import ObjectiveConfig
from wargame_rl.wargame.envs.wargame import WargameEnv


def _two_point_env() -> WargameEnv:
    config = WargameEnvConfig(
        number_of_wargame_models=2,
        number_of_opponent_models=0,
        number_of_objectives=2,
        objective_radius_size=3,
        board_width=40,
        board_height=30,
        objectives=[ObjectiveConfig(x=10, y=15), ObjectiveConfig(x=30, y=15)],
        render_mode=None,
    )
    env = WargameEnv(config)
    env.reset(seed=0)
    return env


def _context(env: WargameEnv) -> StepContext:
    cache = compute_distances(env.wargame_models, env.objectives)
    return StepContext(
        distance_cache=cache,
        current_turn=env.current_turn,
        max_turns=env.max_turns,
        board_width=env.board_width,
        board_height=env.board_height,
    )


def _place(env: WargameEnv, positions: list[tuple[float, float]]) -> None:
    for model, (x, y) in zip(env.wargame_models, positions):
        model.location = np.array([x, y], dtype=float)


def test_one_model_on_each_point_succeeds_and_both_on_one_point_does_not() -> None:
    env = _two_point_env()
    criteria = AllObjectivesOccupiedCriteria()
    _place(env, [(10.0, 15.0), (30.0, 15.0)])
    assert criteria.is_successful(env, _context(env)) is True
    # Both models on the first point: every MODEL is on an objective, which
    # `all_at_objectives` would call success, but the second point is empty.
    _place(env, [(10.0, 15.0), (11.0, 15.0)])
    assert criteria.is_successful(env, _context(env)) is False


def test_a_dead_model_occupies_nothing() -> None:
    env = _two_point_env()
    _place(env, [(10.0, 15.0), (30.0, 15.0)])
    env.wargame_models[1].stats["current_wounds"] = 0
    assert AllObjectivesOccupiedCriteria().is_successful(env, _context(env)) is False


def test_min_models_raises_the_bar_per_point() -> None:
    env = _two_point_env()
    _place(env, [(10.0, 15.0), (30.0, 15.0)])
    assert (
        AllObjectivesOccupiedCriteria(min_models=2).is_successful(env, _context(env))
        is False
    )
    with pytest.raises(ValueError):
        AllObjectivesOccupiedCriteria(min_models=0)


def test_registered_by_name() -> None:
    assert "all_objectives_occupied" in CRITERIA_REGISTRY
    built = build_criteria("all_objectives_occupied", {"min_models": 1})
    assert isinstance(built, AllObjectivesOccupiedCriteria)


def test_an_episode_terminates_on_it_when_configured() -> None:
    """Wired through the phase manager: `terminate_on_success` ends the
    episode on the step every point is occupied."""
    config = WargameEnvConfig(
        number_of_wargame_models=2,
        number_of_opponent_models=0,
        number_of_objectives=2,
        objective_radius_size=3,
        board_width=40,
        board_height=30,
        objectives=[ObjectiveConfig(x=10, y=15), ObjectiveConfig(x=30, y=15)],
        reward_phases=[
            RewardPhaseConfig(
                name="spread",
                terminate_on_success=True,
                reward_calculators=[
                    RewardCalculatorConfig(type="objective_coverage", weight=1.0)
                ],
                success_criteria=SuccessCriteriaConfig(
                    type="all_objectives_occupied", params={}
                ),
            )
        ],
        render_mode=None,
    )
    env = WargameEnv(config)
    env.reset(seed=0)
    _place(env, [(10.0, 15.0), (30.0, 15.0)])
    stay = WargameEnvAction(actions=[0] * len(env.wargame_models))
    _, _, terminated, _, _ = env.step(stay)
    assert terminated is True
