"""The charge decode: the joint charge move, for units the policy declared.

⚠ **A no-op in four phases needs a test PER PHASE.** The reallocation decode
shipped without its movement-phase guard while three `env.step` tests passed —
all three exercised the one phase it acts in, so none could see it overwriting
the shooting slice and the charge ladder (measured at −16 to −24 vp). These
tests assert the no-op in every phase the decode must not touch.
"""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.types.config.battle import OpponentPolicyConfig
from wargame_rl.wargame.envs.types.config.entities import ModelConfig
from wargame_rl.wargame.envs.types.config.melee import MeleeConfig
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.charge_decode import apply_charge_decode
from wargame_rl.wargame.model.common.factory import create_environment

STAY = 0


def _env(melee: bool = True) -> WargameEnv:
    config = WargameEnvConfig(
        number_of_wargame_models=4,
        number_of_opponent_models=4,
        number_of_objectives=1,
        max_groups=1,
        base_radius=0.0,
        melee=MeleeConfig(enabled=melee),
        opponent_policy=OpponentPolicyConfig(
            type="scripted_baseline", params={"baseline": "hold_deployment"}
        ),
        models=[ModelConfig(group_id=0) for _ in range(4)],
        opponent_models=[ModelConfig(group_id=0) for _ in range(4)],
        skip_phases=[] if melee else [BattlePhase.command, BattlePhase.charge],
    )
    env = create_environment(config)
    env.reset(seed=7)
    return env


def _to_phase(env: WargameEnv, phase: BattlePhase) -> None:
    for _ in range(40):
        if env.game_clock_state.phase is phase:
            return
        env.step(WargameEnvAction(actions=[STAY] * len(env.wargame_models)))
    raise AssertionError(f"never reached {phase}")


def _face_off(env: WargameEnv) -> None:
    """Player squad 2" from an enemy squad: a charge is reachable."""
    for index, model in enumerate(env.wargame_models):
        model.location = np.array([10.0, 10.0 + index]).astype(model.location.dtype)
    for index, model in enumerate(env.opponent_models):
        model.location = np.array([12.0, 10.0 + index]).astype(model.location.dtype)


def _declare_all(env: WargameEnv) -> None:
    for model in env.wargame_models:
        model.declared_charge = True
        model.charge_roll = 12.0


@pytest.mark.parametrize(
    "phase", [BattlePhase.command, BattlePhase.movement, BattlePhase.shooting]
)
def test_it_is_a_no_op_outside_the_charge_phase(phase: BattlePhase) -> None:
    """The guard the reallocation decode shipped without."""
    env = _env()
    try:
        _to_phase(env, phase)
        _face_off(env)
        _declare_all(env)
        actions = [STAY] * len(env.wargame_models)

        assert apply_charge_decode(actions, env) == actions
    finally:
        env.close()


def test_it_is_a_no_op_with_melee_off() -> None:
    env = _env(melee=False)
    try:
        actions = [STAY] * len(env.wargame_models)
        assert apply_charge_decode(actions, env) == actions
    finally:
        env.close()


def test_it_is_a_no_op_for_units_that_did_not_declare() -> None:
    """The decode executes the policy's choice; it never makes the choice."""
    env = _env()
    try:
        _to_phase(env, BattlePhase.charge)
        _face_off(env)
        for model in env.wargame_models:
            model.declared_charge = False
        actions = [STAY] * len(env.wargame_models)

        assert apply_charge_decode(actions, env) == actions
    finally:
        env.close()


def test_a_declared_unit_is_moved_rigidly_and_reaches_contact() -> None:
    """Every member gets the SAME action, and the charge stands through env.step."""
    env = _env()
    try:
        _to_phase(env, BattlePhase.charge)
        _face_off(env)
        _declare_all(env)

        decoded = apply_charge_decode([STAY] * len(env.wargame_models), env)

        moved = [a for a in decoded if a != STAY]
        assert moved, "the decode proposed no charge on a reachable board"
        # Rigid: one shared action, which is what preserves coherency exactly.
        assert len(set(moved)) == 1

        env.step(WargameEnvAction(actions=decoded))
        assert any(
            getattr(m, "charged_this_turn", False) for m in env.wargame_models
        ), "the decoded charge did not survive the env's own referee"
    finally:
        env.close()
