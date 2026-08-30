"""`charge_target_binds`: the declared hunt target IS the charge target.

The rules' charge (11-charge-phase.md) selects targets and must end engaged
with all of them and with no non-target. The shipped referee derives its
target from wherever the unit lands, which cannot catch charging A and
landing on B — measured by the fold verdict's mismatch census at 0.65–0.88
of stood charges (§40). Under the flag the declaration binds: the grant
requires the DECLARED unit to be roll-reachable, and the referee reverts a
charge that ends engaged with anything but the declared unit. Everything is
env.step-driven — the fall-back referee's defect lived exactly in the gap
between unit tests and `env.step`.
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


def _env(binds: bool) -> WargameEnv:
    config = WargameEnvConfig(
        number_of_wargame_models=2,
        number_of_opponent_models=4,
        number_of_objectives=1,
        objective_budget=6,
        max_groups=2,
        base_radius=0.0,
        melee=MeleeConfig(enabled=True),
        opponent_policy=OpponentPolicyConfig(
            type="scripted_baseline", params={"baseline": "hold_deployment"}
        ),
        models=[ModelConfig(group_id=0) for _ in range(2)],
        opponent_models=[ModelConfig(group_id=i // 2) for i in range(4)],
        declare_objectives=True,
        declare_targets=True,
        hunt_declares_charge=True,
        charge_target_binds=binds,
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


def _place(env: WargameEnv, declared_group_near: bool) -> None:
    """Player squad with a NEAR bystander (opponent group 0, 2.5" east).

    Opponent group 1 sits 2" north when `declared_group_near` (reachable on
    any 2D6) and 28" east otherwise (beyond the 12" roll cap).
    """
    env.wargame_models[0].location = np.array([10.0, 10.0])
    env.wargame_models[1].location = np.array([10.0, 11.0])
    env.opponent_models[0].location = np.array([12.5, 10.0])
    env.opponent_models[1].location = np.array([12.5, 11.0])
    if declared_group_near:
        env.opponent_models[2].location = np.array([10.0, 13.0])
        env.opponent_models[3].location = np.array([11.0, 13.0])
    else:
        env.opponent_models[2].location = np.array([38.0, 10.0])
        env.opponent_models[3].location = np.array([38.0, 11.0])


def _declare_hunt_of_group_1(env: WargameEnv) -> None:
    hunt = env.player_action_handler.charge_target_slice
    assert hunt is not None
    actions = [0] * len(env.wargame_models)
    actions[0] = hunt.start + 1
    env.step(WargameEnvAction(actions=actions))


def test_a_hunt_of_an_unreachable_group_is_not_granted_despite_a_near_bystander() -> (
    None
):
    """The mismatch grant, refused: reach must be to the DECLARED unit."""
    env = _env(binds=True)
    try:
        _to_phase(env, BattlePhase.command)
        _place(env, declared_group_near=False)
        _declare_hunt_of_group_1(env)
        assert all(int(m.declared_target) == 1 for m in env.wargame_models)
        assert not any(m.declared_charge for m in env.wargame_models)
    finally:
        env.close()


def test_the_same_hunt_is_granted_without_the_flag_which_is_the_defect() -> None:
    """The control: any-enemy reach grants a charge that can only land on
    the bystander -- the behaviour the mismatch census priced."""
    env = _env(binds=False)
    try:
        _to_phase(env, BattlePhase.command)
        _place(env, declared_group_near=False)
        _declare_hunt_of_group_1(env)
        assert all(int(m.declared_target) == 1 for m in env.wargame_models)
        assert all(m.declared_charge for m in env.wargame_models)
    finally:
        env.close()


def test_a_hunt_of_a_reachable_group_is_granted() -> None:
    """The positive control: target-gating is a gate, not a shut valve."""
    env = _env(binds=True)
    try:
        _to_phase(env, BattlePhase.command)
        _place(env, declared_group_near=True)
        _declare_hunt_of_group_1(env)
        assert all(m.declared_charge for m in env.wargame_models)
    finally:
        env.close()


def _charge_at_the_bystander(env: WargameEnv) -> None:
    """From the command phase (hunt of group 1 declared and granted), walk to
    the charge phase and order both models 2" east -- onto the bystander.

    ⚠ In the charge phase the movement slice decodes to the CHARGE ladder,
    `(s + 1) x (12 / n_speeds)` -- so the lowest speed bin (a 1" normal move)
    is a 2" charge, and asking `best_action_toward` for 2" would land a 4"
    one that overshoots into contact with nobody."""
    _to_phase(env, BattlePhase.charge)
    handler = env.player_action_handler
    action = handler.best_action_toward(1.0, 0.0, max_step_length=1.0)
    env.step(WargameEnvAction(actions=[action] * len(env.wargame_models)))


def test_the_referee_reverts_a_charge_that_lands_on_a_bystander() -> None:
    env = _env(binds=True)
    try:
        _to_phase(env, BattlePhase.command)
        _place(env, declared_group_near=True)
        _declare_hunt_of_group_1(env)
        assert all(m.declared_charge for m in env.wargame_models)
        starts = [m.location.copy() for m in env.wargame_models]

        _charge_at_the_bystander(env)

        # Engaged with group 0, not the declared group 1: the charge did not
        # happen -- whole-unit revert, no Strikes First.
        for model, start in zip(env.wargame_models, starts):
            assert np.allclose(model.location, start)
        assert not any(
            getattr(m, "charged_this_turn", False) for m in env.wargame_models
        )
    finally:
        env.close()


def test_the_derived_referee_lets_the_same_charge_stand() -> None:
    """Flag off, same seed, same orders: landing on the bystander is a legal
    derived-target charge -- which is exactly the divergence the flag closes."""
    env = _env(binds=False)
    try:
        _to_phase(env, BattlePhase.command)
        _place(env, declared_group_near=True)
        _declare_hunt_of_group_1(env)
        starts = [m.location.copy() for m in env.wargame_models]

        _charge_at_the_bystander(env)

        moved = any(
            not np.allclose(m.location, s) for m, s in zip(env.wargame_models, starts)
        )
        assert moved
        assert all(getattr(m, "charged_this_turn", False) for m in env.wargame_models)
    finally:
        env.close()


def test_a_hunt_holder_with_an_unreachable_target_may_not_declare_manually() -> None:
    """Panel M0: the referee binds ANY charge by a unit holding a live
    declaration -- manual route included -- so the manual mask must ask the
    grant's question (can the roll reach the DECLARED unit) or it offers a
    declaration whose only outcome is a guaranteed whole-unit revert."""
    from wargame_rl.wargame.envs.env_components.actions import MOVE_TYPE_CHARGE

    for binds, expected in [(True, False), (False, True)]:
        env = _env(binds=binds)
        try:
            _to_phase(env, BattlePhase.command)
            _place(env, declared_group_near=False)
            _declare_hunt_of_group_1(env)
            _to_phase(env, BattlePhase.command)
            _place(env, declared_group_near=False)
            handler = env.player_action_handler
            legality = handler.declaration_legality(
                env.wargame_models, env.opponent_models
            )
            column = handler._move_types.index(MOVE_TYPE_CHARGE)
            assert bool(legality[:, column].any()) is expected, (
                f"binds={binds}: manual charge declarable={legality[:, column].any()}"
            )
        finally:
            env.close()


def test_a_dead_declared_target_falls_through_to_the_derived_referee() -> None:
    """The rules select targets among ALIVE units; a hunt whose unit died is a
    plan for nothing, and the charge is judged as an undeclared one."""
    env = _env(binds=True)
    try:
        _to_phase(env, BattlePhase.command)
        _place(env, declared_group_near=True)
        _declare_hunt_of_group_1(env)
        assert all(m.declared_charge for m in env.wargame_models)
        # Group 1 dies between the declaration and the charge.
        for m in env.opponent_models[2:]:
            m.stats["current_wounds"] = 0
        starts = [m.location.copy() for m in env.wargame_models]

        _charge_at_the_bystander(env)

        moved = any(
            not np.allclose(m.location, s) for m, s in zip(env.wargame_models, starts)
        )
        assert moved
        assert all(getattr(m, "charged_this_turn", False) for m in env.wargame_models)
    finally:
        env.close()
