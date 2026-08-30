"""The fold: one plan per unit, and a hunt IS charge-intent — env.step-driven.

`hunt_declares_charge` (§35's command-slot lesson mechanised): a unit whose
leader declared an enemy target auto-declares its charge whenever the charge is
legal, without spending the leader's command action; and the plan is ONE
commitment — declaring an objective drops the hunt, declaring a hunt drops the
objective. Off (the default) everything is bit-compatible with the v11
semantics: both declarations coexist and no charge is ever auto-declared.
"""

from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.types.config.battle import OpponentPolicyConfig
from wargame_rl.wargame.envs.types.config.entities import ModelConfig
from wargame_rl.wargame.envs.types.config.melee import MeleeConfig
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.factory import create_environment


def _env(fold: bool = True) -> WargameEnv:
    config = WargameEnvConfig(
        number_of_wargame_models=2,
        number_of_opponent_models=2,
        number_of_objectives=1,
        objective_budget=6,
        max_groups=2,
        base_radius=0.0,
        melee=MeleeConfig(enabled=True),
        opponent_policy=OpponentPolicyConfig(
            type="scripted_baseline", params={"baseline": "hold_deployment"}
        ),
        models=[ModelConfig(group_id=0) for _ in range(2)],
        opponent_models=[ModelConfig(group_id=0) for _ in range(2)],
        declare_objectives=True,
        declare_targets=True,
        hunt_declares_charge=fold,
        skip_phases=[BattlePhase.shooting],
    )
    env = create_environment(config)
    env.reset(seed=3)
    return env


def _stay(env: WargameEnv) -> WargameEnvAction:
    return WargameEnvAction(actions=[0] * len(env.wargame_models))


def _to_phase(env: WargameEnv, phase: BattlePhase) -> None:
    while env.game_clock_state.phase is not phase:
        env.step(_stay(env))


def _pin_close(env: WargameEnv) -> None:
    """Player squad 2" from the enemy: charge-eligible, roll-reachable, unengaged."""
    env.wargame_models[0].location = np.array([10.0, 10.0])
    env.wargame_models[1].location = np.array([10.0, 11.0])
    env.opponent_models[0].location = np.array([12.0, 10.0])
    env.opponent_models[1].location = np.array([12.0, 11.0])


def _declare(env: WargameEnv, action_index: int) -> None:
    actions = [0] * len(env.wargame_models)
    actions[0] = action_index
    env.step(WargameEnvAction(actions=actions))


def test_a_hunt_auto_declares_the_charge_and_persists() -> None:
    # Arrange
    env = _env(fold=True)
    try:
        handler = env.player_action_handler
        hunt = handler.charge_target_slice
        assert hunt is not None
        _to_phase(env, BattlePhase.command)
        _pin_close(env)

        # Act: the leader spends its ONE command action on the hunt itself.
        _declare(env, hunt.start + 0)

        # Assert: the unit is hunting AND its charge is declared — the fold
        # granted the charge without the leader touching move_type.
        assert all(int(m.declared_target) == 0 for m in env.wargame_models)
        assert all(m.declared_charge for m in env.wargame_models)

        # A full round later, STAY keeps the plan and the grant renews.
        _to_phase(env, BattlePhase.command)
        _pin_close(env)
        env.step(_stay(env))
        assert all(int(m.declared_target) == 0 for m in env.wargame_models)
        assert all(m.declared_charge for m in env.wargame_models)
    finally:
        env.close()


def test_declaring_an_objective_drops_the_hunt_and_its_charge() -> None:
    env = _env(fold=True)
    try:
        handler = env.player_action_handler
        hunt = handler.charge_target_slice
        plan = handler.objective_target_slice
        assert hunt is not None and plan is not None
        _to_phase(env, BattlePhase.command)
        _pin_close(env)
        _declare(env, hunt.start + 0)

        # Act: next command phase, the leader re-plans onto an objective.
        _to_phase(env, BattlePhase.command)
        _pin_close(env)
        _declare(env, plan.start + 0)

        # Assert: the plan is ONE commitment — hunt gone, no auto-charge.
        assert all(int(m.declared_target) == -1 for m in env.wargame_models)
        assert all(int(m.declared_objective) == 0 for m in env.wargame_models)
        assert not any(m.declared_charge for m in env.wargame_models)
    finally:
        env.close()


def test_a_hunt_with_no_enemy_in_charge_range_is_not_granted() -> None:
    """RENAMED from `test_out_of_range_hunts_are_not_granted_a_charge` (panel
    A, M8): the old name implied the grant is gated on the DECLARED target's
    range, which the code does not do — the gates are ANY-enemy proximity and
    roll-reach (`charge_eligible_units` / `_roll_reachable_units`), so this
    test passes only because EVERY enemy is far. A hunt of a far group with a
    near bystander IS granted; that mismatch is priced by the fold
    pre-registration's mismatch census, not forbidden by code."""
    env = _env(fold=True)
    try:
        handler = env.player_action_handler
        hunt = handler.charge_target_slice
        assert hunt is not None
        _to_phase(env, BattlePhase.command)
        _pin_close(env)
        # Every enemy beyond the 12" charge range: the hunt stands, the grant
        # does not.
        env.opponent_models[0].location = np.array([40.0, 10.0])
        env.opponent_models[1].location = np.array([40.0, 11.0])

        _declare(env, hunt.start + 0)

        assert all(int(m.declared_target) == 0 for m in env.wargame_models)
        assert not any(m.declared_charge for m in env.wargame_models)
    finally:
        env.close()


def test_fold_off_is_the_v11_semantics_exactly() -> None:
    """declare_targets without the fold: hunts never auto-charge, plans coexist."""
    env = _env(fold=False)
    try:
        handler = env.player_action_handler
        hunt = handler.charge_target_slice
        plan = handler.objective_target_slice
        assert hunt is not None and plan is not None
        _to_phase(env, BattlePhase.command)
        _pin_close(env)
        _declare(env, hunt.start + 0)
        assert not any(m.declared_charge for m in env.wargame_models)

        _to_phase(env, BattlePhase.command)
        _pin_close(env)
        _declare(env, plan.start + 0)

        # Both commitments stand side by side — the pre-fold behaviour.
        assert all(int(m.declared_target) == 0 for m in env.wargame_models)
        assert all(int(m.declared_objective) == 0 for m in env.wargame_models)
    finally:
        env.close()
