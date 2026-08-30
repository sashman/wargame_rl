"""The enemy-target declaration and its execution term — env.step-driven.

The hunt analog of `test_declared_objective.py`: the squad's declared enemy
unit as a first-class action (`ActionHandler.declare_targets`) and the payment
for marching at it (`declared_target_progress`). Every test drives `env.step`.
"""

from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs.reward.calculators.declared_target_progress import (
    DeclaredTargetProgressCalculator,
)
from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.types.config.battle import OpponentPolicyConfig
from wargame_rl.wargame.envs.types.config.entities import ModelConfig
from wargame_rl.wargame.envs.types.config.melee import MeleeConfig
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.factory import create_environment


def _env(declare: bool = True) -> WargameEnv:
    config = WargameEnvConfig(
        number_of_wargame_models=4,
        number_of_opponent_models=4,
        number_of_objectives=2,
        objective_budget=6,
        max_groups=2,
        melee=MeleeConfig(enabled=True),
        opponent_policy=OpponentPolicyConfig(
            type="scripted_baseline", params={"baseline": "hold_deployment"}
        ),
        models=[ModelConfig(group_id=i // 2) for i in range(4)],
        opponent_models=[ModelConfig(group_id=i // 2) for i in range(4)],
        declare_targets=declare,
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


def test_the_leader_declares_the_hunt_and_the_squad_is_bound_and_it_persists() -> None:
    # Arrange
    env = _env()
    try:
        handler = env.player_action_handler
        hunt_slice = handler.charge_target_slice
        assert hunt_slice is not None
        _to_phase(env, BattlePhase.command)
        actions = [0] * len(env.wargame_models)
        actions[0] = hunt_slice.start + 1  # squad 0's leader hunts enemy unit 1

        # Act
        env.step(WargameEnvAction(actions=actions))

        # Assert: the whole of squad 0 is bound; squad 1 holds no target.
        for model in env.wargame_models:
            expected = 1 if int(model.group_id) == 0 else -1
            assert int(model.declared_target) == expected

        # A full round of STAY later, the hunt persists (STAY keeps the plan).
        _to_phase(env, BattlePhase.command)
        env.step(_stay(env))
        for model in env.wargame_models:
            if int(model.group_id) == 0:
                assert int(model.declared_target) == 1
    finally:
        env.close()


def test_a_wiped_enemy_unit_is_masked_off_and_a_redeclared_hunt_replaces() -> None:
    # Arrange
    env = _env()
    try:
        handler = env.player_action_handler
        hunt_slice = handler.charge_target_slice
        assert hunt_slice is not None
        for model in env.opponent_models:
            if int(model.group_id) == 1:
                model.stats["current_wounds"] = 0

        # Act
        legality = env.player_charge_target_legality

        # Assert: group 0 declarable, group 1 (wiped) not, for every alive model.
        assert legality.shape == (4, hunt_slice.size)
        assert bool(legality[:, 0].all())
        assert not bool(legality[:, 1].any())

        # Redeclaring replaces: hunt 0, then hunt... 0 is the only legal one,
        # so declare it twice and check the second write is idempotent.
        _to_phase(env, BattlePhase.command)
        actions = [0] * len(env.wargame_models)
        actions[0] = hunt_slice.start
        env.step(WargameEnvAction(actions=actions))
        assert int(env.wargame_models[1].declared_target) == 0
    finally:
        env.close()


def test_marching_at_the_declared_unit_pays_and_hovering_does_not() -> None:
    # Arrange: real env.step contexts, exactly as the phase manager feeds them.
    env = _env()
    try:
        handler = env.player_action_handler
        hunt_slice = handler.charge_target_slice
        assert hunt_slice is not None
        calculator = DeclaredTargetProgressCalculator(value=1.0, span=6.0)
        calculator.reset_episode()
        # Put model 0 west of enemy group 0's centroid and declare the hunt.
        enemy = [m for m in env.opponent_models if int(m.group_id) == 0]
        centroid = np.mean([np.asarray(m.location, dtype=float) for m in enemy], axis=0)
        env.wargame_models[0].location = (centroid - [10.0, 0.0]).astype(
            env.wargame_models[0].location.dtype
        )
        _to_phase(env, BattlePhase.command)
        actions = [0] * len(env.wargame_models)
        actions[0] = hunt_slice.start
        env.step(WargameEnvAction(actions=actions))
        seed_ctx = env.last_step_context
        assert seed_ctx is not None
        calculator.calculate(0, env.wargame_models[0], env, seed_ctx)

        # Act: march model 0 due east, at the declared unit.
        assert env.game_clock_state.phase is BattlePhase.movement
        grid = handler._displacements.reshape(-1, 2)
        east = int(np.argmax(grid[:, 0] - np.abs(grid[:, 1])))
        move = [0] * len(env.wargame_models)
        move[0] = handler.movement_slice.start + east
        before = env.wargame_models[0].location.copy()
        env.step(WargameEnvAction(actions=move))
        moved = float(np.linalg.norm(env.wargame_models[0].location - before))
        ctx = env.last_step_context
        assert ctx is not None
        paid_march = calculator.calculate(0, env.wargame_models[0], env, ctx)

        # Next round, same phase, STAY: hovering pays nothing.
        _to_phase(env, BattlePhase.movement)
        env.step(_stay(env))
        ctx2 = env.last_step_context
        assert ctx2 is not None
        paid_hover = calculator.calculate(0, env.wargame_models[0], env, ctx2)

        # Assert
        assert moved > 0.0, "the fixture's march did not move"
        assert paid_march > 0.0, "closing on the declared unit paid nothing"
        assert paid_hover == 0.0, "standing still re-earned the payment"
    finally:
        env.close()


def test_the_flag_off_is_a_no_op_and_registers_no_slice() -> None:
    # Arrange / Act
    env = _env(declare=False)
    try:
        # Assert
        assert env.player_action_handler.charge_target_slice is None
        assert env.player_charge_target_legality.shape[1] == 0
        _to_phase(env, BattlePhase.command)
        env.step(_stay(env))
        for model in env.wargame_models:
            assert int(getattr(model, "declared_target", -1)) == -1
    finally:
        env.close()
