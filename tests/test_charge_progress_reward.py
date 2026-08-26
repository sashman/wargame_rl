"""`charge_progress`: a gradient toward a feasible charge, where the referee gives none.

`_enforce_charge` is all-or-nothing — a unit that ends coherent and engaged with
exactly one enemy unit keeps its move, and one that misses by a hair is put back
where it started. So a near-miss and a wild miss produce identical feedback, and
the only signal a policy gets about charging is one it almost never observes.

⚠ Measured: three clones of a charging teacher reproduced its shooting at 0.99
and echoed **0.8–2.4%** of its charge orders. The bottleneck is *proposing* a
rare all-or-nothing action, which is what this term is for.
"""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.envs.env_components.distance_cache import compute_distances
from wargame_rl.wargame.envs.reward.calculators.charge_progress import (
    ChargeProgressCalculator,
)
from wargame_rl.wargame.envs.reward.step_context import StepContext
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.types.config import MeleeConfig, MeleeWeaponProfile
from wargame_rl.wargame.envs.types.config.battle import OpponentPolicyConfig
from wargame_rl.wargame.envs.types.config.entities import ModelConfig
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.factory import create_environment

VALUE = 0.05


def _env(melee: bool = True) -> WargameEnv:
    config = WargameEnvConfig(
        number_of_wargame_models=2,
        number_of_opponent_models=2,
        number_of_objectives=1,
        opponent_policy=OpponentPolicyConfig(
            type="scripted_baseline", params={"baseline": "hold_deployment"}
        ),
        models=[
            ModelConfig(group_id=0, melee_weapons=[MeleeWeaponProfile()])
            for _ in range(2)
        ],
        opponent_models=[
            ModelConfig(group_id=0, melee_weapons=[MeleeWeaponProfile()])
            for _ in range(2)
        ],
        melee=MeleeConfig(enabled=melee),
        engagement_range=1.0,
        base_radius=0.0,
        # ⚠ A stepped charge phase needs a stepped COMMAND phase: the charge
        # is declared there by the unit's leader. Skipping charge is what lets
        # the melee-off case keep command skipped.
        skip_phases=(
            [BattlePhase.shooting, BattlePhase.fight]
            if melee
            else [
                BattlePhase.command,
                BattlePhase.shooting,
                BattlePhase.charge,
                BattlePhase.fight,
            ]
        ),
    )
    env = create_environment(config)
    env.reset(seed=5)
    return env


def _ctx(env: WargameEnv) -> StepContext:
    return StepContext(
        distance_cache=compute_distances(env.wargame_models, env.objectives),
        current_turn=env.current_turn,
        max_turns=env.max_turns,
        board_width=env.board_width,
        board_height=env.board_height,
    )


def _pay(
    env: WargameEnv, gap: float, roll: float = 6.0, declared: bool = True
) -> float:
    """This model's payment with the nearest living enemy `gap` inches away."""
    calculator = ChargeProgressCalculator(value=VALUE)
    player = env.wargame_models[0]
    player.location = np.array([10.0, 10.0], dtype=player.location.dtype)
    for index, model in enumerate(env.opponent_models):
        model.location = np.array(
            [10.0 + gap + index * 40.0, 10.0], dtype=model.location.dtype
        )
    for model in env.wargame_models:
        model.charge_roll = roll
        model.declared_charge = declared
    calculator.reset_episode()
    return calculator.calculate(0, player, env, _ctx(env))


def _charge_phase(env: WargameEnv) -> None:
    while env.game_clock_state.phase is not BattlePhase.charge:
        env.step(env.action_space.sample() and _stay(env))


def _stay(env: WargameEnv):  # type: ignore[no-untyped-def]
    from wargame_rl.wargame.envs.types import WargameEnvAction

    return WargameEnvAction(actions=[0] * len(env.wargame_models))


def test_it_pays_nothing_outside_the_charge_phase() -> None:
    """The gate that separates this from a travel reward.

    `closest_objective_v2` is the four-times-refuted "walk toward the thing"
    term, and a teleport audit priced walking a squad at defended ground at
    −29.4 of its own income. This pays only where the charge decision is made.
    """
    # Arrange
    env = _env()
    try:
        assert env.game_clock_state.phase is not BattlePhase.charge

        # Act / Assert
        assert _pay(env, gap=2.0) == 0.0
    finally:
        env.close()


def test_it_pays_nothing_when_melee_is_off() -> None:
    """The no-op guarantee: every golden config must be untouched."""
    # Arrange
    env = _env(melee=False)
    try:
        # Act / Assert
        assert _pay(env, gap=2.0) == 0.0
    finally:
        env.close()


def test_closing_pays_MORE_which_is_the_whole_gradient() -> None:
    """A near-miss must be worth more than a wild miss — the referee says neither."""
    # Arrange
    env = _env()
    _charge_phase(env)
    try:
        # Act
        far = _pay(env, gap=11.0)
        near = _pay(env, gap=3.0)
        touching = _pay(env, gap=0.5)

        # Assert
        assert far < near < touching
    finally:
        env.close()


def test_it_is_MAXIMAL_when_engaged_so_landing_beats_missing() -> None:
    """Nothing may pay more for missing than for hitting.

    Contact is base to base, exactly as the engagement predicate measures it —
    a centre-to-centre reading would make the maximum unreachable.
    """
    # Arrange
    env = _env()
    _charge_phase(env)
    try:
        # Act / Assert
        assert _pay(env, gap=0.5) == pytest.approx(VALUE)
        assert _pay(env, gap=0.0) == pytest.approx(VALUE)
    finally:
        env.close()


def test_a_unit_the_rules_forbid_to_charge_is_paid_nothing() -> None:
    """Advancing or falling back spends the charge, and a roll of zero is no roll."""
    # Arrange
    env = _env()
    _charge_phase(env)
    try:
        assert _pay(env, gap=3.0) > 0.0

        # Act / Assert
        assert _pay(env, gap=3.0, roll=0.0) == 0.0
        for model in env.wargame_models:
            model.fell_back_this_turn = True
        assert _pay(env, gap=3.0) == 0.0
    finally:
        env.close()


def test_a_dead_model_earns_nothing() -> None:
    """`phase_manager` iterates alive models; this must agree with it."""
    # Arrange
    env = _env()
    _charge_phase(env)
    try:
        calculator = ChargeProgressCalculator(value=VALUE)
        player = env.wargame_models[0]
        player.stats["current_wounds"] = 0

        # Act / Assert
        assert calculator.calculate(0, player, env, _ctx(env)) == 0.0
    finally:
        env.close()


def test_a_unit_that_did_NOT_declare_is_paid_NOTHING() -> None:
    """The regression test for the gate that was missing, and the whole term.

    ⚠ Shipped 2026-08-25 without this gate: the loop tested `charge_roll <= 0.0`
    and called it the declaration check, but `_roll_charge_dice` rolls 2D6 for
    every unit of the active side unconditionally, so it excluded nothing. The
    term paid every alive model for closing on the nearest enemy. Measured on
    the shipped melee config, `squad_march_take` -- which declares ZERO charges
    -- earned **5.713 per episode with 0.0% of it reaching a declared unit**,
    against `squad_march_take_charge`'s 4.196: the term paid the non-charging
    policy **36% more** than the charging one.

    ⚠ Six tests covered this module and none set `declared_charge`, so the
    suite asserted the term against its own relaxation -- verbatim the defect
    this project already paid +11.4 vp for on the joint decoder.
    """
    # Arrange
    env = _env()
    _charge_phase(env)
    try:
        # Act
        declared = _pay(env, gap=3.0, declared=True)
        undeclared = _pay(env, gap=3.0, declared=False)

        # Assert
        assert declared > 0.0
        assert undeclared == 0.0
    finally:
        env.close()


def test_the_declaration_it_gates_on_is_OBSERVABLE() -> None:
    """A term keyed on state the network cannot see is the standing check.

    ⚠ CLAUDE.md: "Check the agent can OBSERVE what the lever keys on. A desk
    check that costs seconds and has burned ~10 GPU-hours." Fixing the gate
    without this would have moved the term from paying the wrong behaviour to
    paying for something invisible.
    """
    # Arrange
    env = _env()
    try:
        # Act
        observation = env._get_obs()

        # Assert
        assert observation.wargame_models[0].declared_charge is not None
        assert observation.opponent_models[0].declared_charge is not None
    finally:
        env.close()
