"""Tests for the two per-model calculators added for credit assignment.

Only 3 of 8 existing calculators were per-model, and both of the useful ones go
silent once models arrive — `closest_objective`'s progress is a potential that
exhausts on arrival and is exactly 0 on shooting steps, and `group_cohesion`
returns a hard 0 inside its limit. So all 25 models shared an identical reward
for most of an episode, which defeats per-model credit assignment.

`objective_hold` fires while stationary; `model_kills` gives the shooting head
a credit path it did not have.
"""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.envs.env_components.distance_cache import compute_distances
from wargame_rl.wargame.envs.reward.calculators.model_kills import ModelKillsCalculator
from wargame_rl.wargame.envs.reward.calculators.objective_hold import (
    ObjectiveHoldCalculator,
)
from wargame_rl.wargame.envs.reward.calculators.registry import build_calculator
from wargame_rl.wargame.envs.reward.step_context import StepContext
from wargame_rl.wargame.envs.types import (
    ModelConfig,
    ObjectiveConfig,
    OpponentPolicyConfig,
    WargameEnvAction,
    WargameEnvConfig,
)
from wargame_rl.wargame.envs.types.config import WeaponProfile
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv

OBJECTIVE = (20, 20)


def _make_env(n_player_on_objective: int, n_opponent_on_objective: int) -> WargameEnv:
    """Env with a controllable number of models standing on one objective.

    Models not on the objective are parked far away in their own corner.
    """
    player = [
        ModelConfig(x=OBJECTIVE[0], y=OBJECTIVE[1], group_id=0)
        for _ in range(n_player_on_objective)
    ] + [ModelConfig(x=2, y=2, group_id=0)]
    opponent = [
        ModelConfig(x=OBJECTIVE[0], y=OBJECTIVE[1], group_id=0)
        for _ in range(n_opponent_on_objective)
    ] + [ModelConfig(x=38, y=38, group_id=0)]

    config = WargameEnvConfig(
        render_mode=None,
        board_width=40,
        board_height=40,
        number_of_wargame_models=len(player),
        number_of_opponent_models=len(opponent),
        number_of_objectives=1,
        objective_radius_size=2,
        number_of_battle_rounds=4,
        models=player,
        opponent_models=opponent,
        objectives=[ObjectiveConfig(x=OBJECTIVE[0], y=OBJECTIVE[1])],
        opponent_policy=OpponentPolicyConfig(type="scripted_advance_to_objective"),
    )
    env = WargameEnv(config=config)
    env.reset(seed=0)
    return env


def _context(env: WargameEnv, **kwargs: object) -> StepContext:
    return StepContext(
        distance_cache=compute_distances(env.wargame_models, env.objectives),
        current_turn=1,
        max_turns=env.max_turns,
        board_width=env.board_width,
        board_height=env.board_height,
        **kwargs,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------
# objective_hold
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_player", "n_opponent", "expected"),
    [
        (3, 1, 1.0),  # player-controlled: strictly more models
        (2, 2, 0.5),  # contested: equal counts, nobody scores
        (1, 3, 0.25),  # opponent-held
    ],
)
def test_value_scales_with_control_state(
    n_player: int, n_opponent: int, expected: float
) -> None:
    """A model on an objective is paid by who controls it, not merely by standing.

    This is what makes it dominate `models_at_objectives`, which every
    competent baseline saturates at 1.0 and so cannot rank policies.
    """
    env = _make_env(n_player, n_opponent)
    calculator = ObjectiveHoldCalculator()

    value = calculator.calculate(0, env.wargame_models[0], env, _context(env))

    assert value == pytest.approx(expected)


def test_model_off_the_objective_scores_nothing() -> None:
    """Only models actually standing on an objective are paid."""
    env = _make_env(3, 1)
    calculator = ObjectiveHoldCalculator()
    off_objective_index = len(env.wargame_models) - 1

    value = calculator.calculate(
        off_objective_index, env.wargame_models[off_objective_index], env, _context(env)
    )

    assert value == pytest.approx(0.0)


def test_values_differ_across_models_in_the_same_step() -> None:
    """The whole point: the reward vector is not constant across models."""
    env = _make_env(3, 1)
    calculator = ObjectiveHoldCalculator()
    ctx = _context(env)

    values = [
        calculator.calculate(i, model, env, ctx)
        for i, model in enumerate(env.wargame_models)
    ]

    assert len(set(values)) > 1


def test_control_state_is_computed_once_per_step() -> None:
    """The opponent cache is built once a step, not once per model.

    `calculate` runs per model and deriving control needs opponent distances,
    so without caching a 25-model army would rebuild it 25 times a step.
    """
    env = _make_env(3, 1)
    calculator = ObjectiveHoldCalculator()
    ctx = _context(env)

    calculator.calculate(0, env.wargame_models[0], env, ctx)
    cached = calculator._cached_values
    calculator.calculate(1, env.wargame_models[1], env, ctx)

    assert calculator._cached_values is cached


# --------------------------------------------------------------------------
# model_kills
# --------------------------------------------------------------------------


def test_kills_are_credited_to_the_model_that_made_them() -> None:
    """A model that killed twice scores twice; one that fired and missed scores 0.

    Under the global `killing` calculator every model receives the same value
    regardless of who shot, so this is the assertion that fails today.
    """
    env = _make_env(3, 1)
    calculator = ModelKillsCalculator(bonus_per_kill=2.0)
    kills = np.zeros(len(env.wargame_models), dtype=np.int64)
    kills[0] = 2

    ctx = _context(env, player_kills_by_model=kills)
    scores = [
        calculator.calculate(i, model, env, ctx)
        for i, model in enumerate(env.wargame_models)
    ]

    assert scores[0] == pytest.approx(4.0)
    assert all(s == pytest.approx(0.0) for s in scores[1:])


def test_no_shooting_resolved_scores_nothing() -> None:
    """A step with no shooting pays nothing rather than erroring."""
    env = _make_env(3, 1)
    calculator = ModelKillsCalculator()

    value = calculator.calculate(0, env.wargame_models[0], env, _context(env))

    assert value == pytest.approx(0.0)


def _make_shooting_env() -> WargameEnv:
    """Armed player models in range of, but not engaged with, the opponent.

    Weapons are required (`ModelConfig.weapons` defaults to empty, which means
    a model cannot shoot at all), and the two sides must be more than
    `ENGAGEMENT_RANGE` apart or the shooting mask forbids firing.
    """
    weapon = WeaponProfile(range=12, attacks=2)
    config = WargameEnvConfig(
        render_mode=None,
        board_width=40,
        board_height=40,
        number_of_wargame_models=3,
        number_of_opponent_models=3,
        number_of_objectives=1,
        objective_radius_size=2,
        number_of_battle_rounds=6,
        # The default skips every non-movement phase, so the shooting phase
        # would never execute and no shot could resolve.
        skip_phases=[BattlePhase.command, BattlePhase.charge, BattlePhase.fight],
        models=[
            ModelConfig(x=10, y=20 + i, group_id=0, weapons=[weapon]) for i in range(3)
        ],
        opponent_models=[ModelConfig(x=18, y=20 + i, group_id=0) for i in range(3)],
        objectives=[ObjectiveConfig(x=OBJECTIVE[0], y=OBJECTIVE[1])],
        opponent_policy=OpponentPolicyConfig(type="scripted_advance_to_objective"),
    )
    env = WargameEnv(config=config)
    env.reset(seed=0)
    return env


def test_env_attributes_kills_to_the_firing_model() -> None:
    """End-to-end: the per-model kill vector always sums to the scalar count.

    Exercises the real shooting path rather than a hand-built context, so it
    covers the `PairedShootingResult.killed` flag and the wiring in `step`.
    """
    env = _make_shooting_env()
    total_kills = 0
    per_model_total = np.zeros(len(env.wargame_models), dtype=np.int64)
    shooting_slice = env.player_action_handler.shooting_slice
    assert shooting_slice is not None

    for episode in range(6):
        env.reset(seed=episode)
        terminated = truncated = False
        while not (terminated or truncated):
            # Fire deliberately rather than sampling: a random action lands in
            # the shooting slice only ~3% of the time, which would make this
            # test pass or fail on the dice.
            if env.game_clock_state.phase is BattlePhase.shooting:
                actions = [shooting_slice.start] * len(env.wargame_models)
            else:
                actions = [0] * len(env.wargame_models)
            _obs, _r, terminated, truncated, _info = env.step(
                WargameEnvAction(actions=actions)
            )
            ctx = env.last_step_context
            assert ctx is not None and ctx.player_kills_by_model is not None
            per_model_total += ctx.player_kills_by_model
            total_kills += ctx.player_models_killed

    # Guard against a vacuous pass: with no weapons, or with both sides inside
    # engagement range, no shot ever resolves and the assertion below is 0 == 0.
    assert total_kills > 0
    assert int(per_model_total.sum()) == total_kills


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["objective_hold", "model_kills"])
def test_calculators_are_registered_and_per_model(name: str) -> None:
    """Both are buildable by name and carry per-model credit."""
    from wargame_rl.wargame.envs.reward.calculators.base import PerModelRewardCalculator

    calculator = build_calculator(name, weight=1.0, params={})

    assert isinstance(calculator, PerModelRewardCalculator)
