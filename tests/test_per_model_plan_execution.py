"""The plan-weighted execution reward, #384's execution phase: the task
weights follow the environment's arrival rule, a member is paid the weighted
task terms keyed to its committed objective and nothing else, leaving in hold
mode pays exactly zero, and the close's outcome terms go to the planning
stream. `execution: keyed` (the default) is untouched.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.per_model_seats import random_legal_action, small_config
from wargame_rl.wargame.envs.per_model import PerModelEnv
from wargame_rl.wargame.envs.per_model.commitment import NO_TARGET, retire_and_reassign
from wargame_rl.wargame.envs.per_model.reward_timing import (
    TASK_APPROACH,
    TASK_HOLD,
    TASK_OF,
    PerStepReward,
)
from wargame_rl.wargame.envs.reward.phase import RewardCalculatorConfig
from wargame_rl.wargame.envs.types import BattlePhase, WargameEnvConfig
from wargame_rl.wargame.envs.types.config import CommitmentConfig


def _plan_config(execution: str = "plan") -> WargameEnvConfig:
    base = small_config()
    phase = base.reward_phases[0].model_copy(
        update={
            "reward_calculators": [
                RewardCalculatorConfig(
                    type="closest_objective_v2",
                    weight=1.0,
                    params={
                        "progress_scale": 6.0,
                        "fallback_to_nearest": True,
                        "overstack_penalty_per_extra": 0.0,
                    },
                ),
                RewardCalculatorConfig(type="objective_stay", weight=1.0),
                RewardCalculatorConfig(type="objective_coverage", weight=0.3),
            ]
        }
    )
    copied: WargameEnvConfig = base.model_copy(
        update={
            "reward_phases": [phase],
            "commitments": CommitmentConfig(assignment="greedy", execution=execution),  # type: ignore[arg-type]
        }
    )
    return copied


def _place_unit(env: PerModelEnv, group: int, location: np.ndarray) -> None:
    for model in env.wargame_models:
        if int(model.group_id) == group:
            model.location = location.copy()


def test_task_weights_follow_the_arrival_rule_and_reset_on_a_new_target() -> None:
    env = PerModelEnv(_plan_config())
    env.reset(seed=1)
    state = env.player_commitments
    g = env.player_seat.living_units()[0]
    k = state.ground_of(g)
    assert k != NO_TARGET
    assert state.task_weights(g) == (1.0, 0.0), "approach until the first arrival"
    _place_unit(env, g, np.asarray(env.objectives[k].location, dtype=float))
    retire_and_reassign(state, env.wargame_models, env.opponent_models, env.objectives)
    assert state.task_weights(g) == (0.0, 1.0), "hold once arrived"
    _place_unit(env, g, np.asarray(env.objectives[k].location, dtype=float) + 20.0)
    retire_and_reassign(state, env.wargame_models, env.opponent_models, env.objectives)
    assert state.task_weights(g) == (0.0, 1.0), "a walk-off does not end the hold"
    other = next(j for j in range(len(env.objectives)) if j != k)
    state.set_ground(g, other)
    assert state.task_weights(g) == (1.0, 0.0), "a new target starts an approach"
    state.clear_ground(g)
    assert state.task_weights(g) == (0.0, 0.0), "no plan, no weight"
    weights = state.task_weights_per_model(env.wargame_models)
    assert weights.shape == (len(env.wargame_models), 2)


def test_tasks_of_the_per_decision_terms() -> None:
    assert TASK_OF["closest_objective_v2"] == TASK_APPROACH
    assert TASK_OF["objective_stay"] == TASK_HOLD
    assert TASK_OF["model_kills"] is None


def _pay_at(env: PerModelEnv, retimer: PerStepReward, index: int, name: str) -> float:
    """What term `name` pays model `index` on a step it acted, under the
    retimer's plan weighting; the calculators are the env's own."""
    ctx = retimer._context(action_phase=BattlePhase.movement, kills_by_model=None)
    calculators = {n: c for n, c in retimer.classes.potential}
    calculator = calculators[name]
    model = env.wargame_models[index]
    raw = calculator.weight * calculator.calculate(index, model, env, ctx)  # type: ignore[arg-type]
    return raw * retimer._task_weight(ctx, name, index)


def test_a_member_is_paid_the_plan_weighted_terms_and_nothing_else() -> None:
    """Approach mode: the travel term pays, the staying term is zeroed even
    inside the objective. Hold mode: the staying term pays inside, the travel
    term is zeroed, and a step that leaves pays exactly zero."""
    env = PerModelEnv(_plan_config())
    retimer = PerStepReward(env, streams=True)
    env.reset(seed=2)
    retimer.reset()
    state = env.player_commitments
    g = env.player_seat.living_units()[0]
    k = state.ground_of(g)
    index = env.player_seat.unit_members(g)[0]
    target = np.asarray(env.objectives[k].location, dtype=float)
    # Start the unit well outside its objective, prime the travel potential
    # there, then step one member closer.
    _place_unit(env, g, target + np.array([20.0, 0.0]))
    _pay_at(env, retimer, index, "closest_objective_v2")
    model = env.wargame_models[index]
    model.location = model.location + (target - model.location) * 0.5
    assert _pay_at(env, retimer, index, "closest_objective_v2") > 0.0
    # Inside the objective before the plan flips: no hold pay yet.
    _place_unit(env, g, target)
    assert _pay_at(env, retimer, index, "objective_stay") == 0.0
    # The close flips the plan to hold.
    retire_and_reassign(state, env.wargame_models, env.opponent_models, env.objectives)
    assert state.task_weights(g) == (0.0, 1.0)
    assert _pay_at(env, retimer, index, "objective_stay") > 0.0
    _pay_at(env, retimer, index, "closest_objective_v2")  # re-anchor at the target
    # Leaving in hold mode: the travel term is weighted 0 and the staying term
    # pays nothing outside -- exactly zero, not a small charge.
    model.location = target + np.array([6.0, 0.0])
    assert _pay_at(env, retimer, index, "closest_objective_v2") == 0.0
    assert _pay_at(env, retimer, index, "objective_stay") == 0.0
    # Standing on someone else's objective in hold mode: nothing.
    other = next(j for j in range(len(env.objectives)) if j != k)
    model.location = np.asarray(env.objectives[other].location, dtype=float)
    assert _pay_at(env, retimer, index, "objective_stay") == 0.0


def test_the_keyed_execution_weights_nothing() -> None:
    env = PerModelEnv(_plan_config(execution="keyed"))
    retimer = PerStepReward(env)
    env.reset(seed=3)
    retimer.reset()
    ctx = retimer._context(action_phase=None, kills_by_model=None)
    assert ctx.task_weights is None
    for name, _ in retimer.classes.potential:
        assert retimer._task_weight(ctx, name, 0) == 1.0
    assert not env.config.commitments.streams


def test_under_plan_execution_the_close_outcome_never_reaches_a_member() -> None:
    config = _plan_config()
    assert config.commitments.streams
    env = PerModelEnv(config)
    retimer = PerStepReward(env, streams=config.commitments.streams)
    observation, _ = env.reset(seed=4)
    retimer.reset()
    rng = np.random.default_rng(4)
    closes = 0
    for _ in range(400):
        action = random_legal_action(observation.decision, rng)
        nxt, _, terminated, _, info = env.step(action)
        payment = retimer.on_step(observation, action, info["effect"], terminated)
        if payment.is_close:
            closes += 1
            coverage = payment.breakdown.get("objective_coverage", 0.0)
            assert payment.planning == pytest.approx(
                coverage
                + sum(
                    v for k, v in payment.breakdown.items() if k.startswith("terminal")
                )
            )
        else:
            assert payment.planning == 0.0
        observation = nxt
        if terminated:
            break
    assert closes >= 2
