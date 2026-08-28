"""The objective declaration and its execution term — env.step-driven.

The agent's own allocation plan as a first-class action
(`ActionHandler.declare_objectives`), and the payment for executing it
(`declared_objective_progress`). Every test drives `env.step`; this project
has paid three times for suites whose unit tests could not see a defect the
step pipeline produced.
"""

from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs.reward.calculators.declared_objective_progress import (
    DeclaredObjectiveProgressCalculator,
)
from wargame_rl.wargame.envs.types import WargameEnvAction, WargameEnvConfig
from wargame_rl.wargame.envs.types.config.battle import OpponentPolicyConfig
from wargame_rl.wargame.envs.types.config.entities import ModelConfig
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.model.common.factory import create_environment


def _env(declare: bool = True) -> WargameEnv:
    config = WargameEnvConfig(
        number_of_wargame_models=4,
        number_of_opponent_models=2,
        number_of_objectives=2,
        objective_budget=6,
        opponent_policy=OpponentPolicyConfig(
            type="scripted_baseline", params={"baseline": "hold_deployment"}
        ),
        models=[ModelConfig(group_id=i // 2) for i in range(4)],
        opponent_models=[ModelConfig(group_id=0) for _ in range(2)],
        declare_objectives=declare,
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


def test_the_leader_declares_and_the_squad_is_bound_and_the_plan_persists() -> None:
    # Arrange
    env = _env()
    try:
        handler = env.player_action_handler
        target_slice = handler.objective_target_slice
        assert target_slice is not None
        _to_phase(env, BattlePhase.command)
        leader = next(
            i
            for i, m in enumerate(env.wargame_models)
            if m.is_alive and m.group_id == env.wargame_models[0].group_id
        )
        group = int(env.wargame_models[leader].group_id)
        actions = [0] * len(env.wargame_models)
        actions[leader] = target_slice.start + 1

        # Act — declare objective 1, then run a full round of STAY.
        env.step(WargameEnvAction(actions=actions))
        for _ in range(8):
            env.step(_stay(env))

        # Assert — the whole squad carries the plan, and it SURVIVED the turn
        # boundary (begin_turn does not clear it).
        declared = {
            int(m.declared_objective)
            for m in env.wargame_models
            if int(m.group_id) == group
        }
        assert declared == {1}, f"the squad's plan is {declared}, not {{1}}"
        others = {
            int(m.declared_objective)
            for m in env.wargame_models
            if int(m.group_id) != group
        }
        assert others == {-1}, "a squad that declared nothing acquired a plan"
    finally:
        env.close()


def test_STAY_keeps_the_plan_and_a_new_declaration_replaces_it() -> None:
    # Arrange
    env = _env()
    try:
        handler = env.player_action_handler
        target_slice = handler.objective_target_slice
        assert target_slice is not None
        _to_phase(env, BattlePhase.command)
        leader = 0
        group = int(env.wargame_models[leader].group_id)
        actions = [0] * len(env.wargame_models)
        actions[leader] = target_slice.start  # objective 0

        # Act
        env.step(WargameEnvAction(actions=actions))
        _to_phase(env, BattlePhase.command)
        env.step(_stay(env))  # STAY: keep
        kept = int(env.wargame_models[leader].declared_objective)
        _to_phase(env, BattlePhase.command)
        actions[leader] = target_slice.start + 1
        env.step(WargameEnvAction(actions=actions))  # re-plan
        replaced = int(env.wargame_models[leader].declared_objective)

        # Assert
        assert kept == 0
        assert replaced == 1
        assert group is not None
    finally:
        env.close()


def test_the_mask_forbids_declaring_a_padding_objective() -> None:
    """Budget 6, episode draws 2 — indices 2..5 are plans for nothing."""
    env = _env()
    try:
        _to_phase(env, BattlePhase.command)
        legality = env.player_objective_target_legality
        assert legality[:, :2].all(), "a real objective was masked off"
        assert not legality[:, 2:].any(), "a padding objective is declarable"
    finally:
        env.close()


def test_the_execution_term_pays_the_march_and_not_the_hover() -> None:
    """Delta semantics through env.step: closing pays, standing still does not."""
    env = _env()
    try:
        handler = env.player_action_handler
        target_slice = handler.objective_target_slice
        assert target_slice is not None
        calculator = DeclaredObjectiveProgressCalculator(value=1.0, span=6.0)
        calculator.reset_episode()
        # Put squad 0 west of objective 0, declare it, and march east.
        objective = np.asarray(env.objectives[0].location, dtype=float)
        for i in range(2):
            env.wargame_models[i].location = (objective - [10.0, -0.3 * i]).astype(
                env.wargame_models[i].location.dtype
            )
        _to_phase(env, BattlePhase.command)
        actions = [0] * len(env.wargame_models)
        actions[0] = target_slice.start
        env.step(WargameEnvAction(actions=actions))
        # Seed the delta's baseline exactly as the phase manager does: the
        # calculator is invoked on EVERY step, so the command step's context
        # provides the pre-march gaps.
        seed_ctx = env.last_step_context
        assert seed_ctx is not None
        calculator.calculate(0, env.wargame_models[0], env, seed_ctx)

        # Movement phase: model 0 marches toward the objective; feed the
        # calculator the contexts the phase manager builds.
        paid_march = 0.0
        paid_hover = 0.0
        assert env.game_clock_state.phase is BattlePhase.movement
        # movement action: pick the movement-slice action closest to due east
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

        # Next round, same phase, STAY: hover pays nothing.
        _to_phase(env, BattlePhase.movement)
        env.step(_stay(env))
        ctx2 = env.last_step_context
        assert ctx2 is not None
        paid_hover = calculator.calculate(0, env.wargame_models[0], env, ctx2)

        # Assert
        assert moved > 0.0, "the fixture's march did not move"
        assert paid_march > 0.0, "closing on the declared objective paid nothing"
        assert paid_hover == 0.0, "standing still re-earned the payment"
    finally:
        env.close()


def test_the_flag_off_is_a_structural_noop() -> None:
    env = _env(declare=False)
    try:
        assert env.player_action_handler.objective_target_slice is None
        obs, _ = env.reset(seed=3)
        assert obs.wargame_models[0].declared_objective_onehot is None
    finally:
        env.close()
