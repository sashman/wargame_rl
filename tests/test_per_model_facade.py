"""The per-model facade's own step semantics: the unit lock, declarations
that consume a unit, the turn-closing step, and the two clocks."""

from __future__ import annotations

import numpy as np
import pytest

from wargame_rl.wargame.envs.per_model import (
    MoveDeclaration,
    PerModelAction,
    PerModelEnv,
    StepKind,
)
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.types.game_timing import BattlePhase


def _movement_only_config(n_rounds: int = 3) -> WargameEnvConfig:
    return WargameEnvConfig(
        render_mode=None,
        number_of_wargame_models=8,
        max_groups=2,
        number_of_objectives=2,
        number_of_battle_rounds=n_rounds,
    )


def _fresh(n_rounds: int = 3) -> PerModelEnv:
    env = PerModelEnv(_movement_only_config(n_rounds), build_info=False)
    env.reset(seed=3)
    return env


def test_the_unit_lock_binds_after_the_first_member_acts() -> None:
    env = _fresh()
    observation = env._observe()
    assert observation.kind is StepKind.model_action
    assert observation.selection_mask.all(), "every alive model starts selectable"

    unit_of_five = int(env.wargame_models[5].group_id)
    observation, *_ = env.step(PerModelAction(model_index=5))
    assert observation.open_unit == unit_of_five
    legal = np.flatnonzero(observation.selection_mask)
    assert all(int(env.wargame_models[i].group_id) == unit_of_five for i in legal), (
        "while a unit is open, only its own un-acted members are legal"
    )
    assert 5 not in legal


def test_the_mask_widens_when_the_unit_closes() -> None:
    env = _fresh()
    members = [
        i
        for i, m in enumerate(env.wargame_models)
        if int(m.group_id) == int(env.wargame_models[0].group_id)
    ]
    observation = env._observe()
    for index in members:
        observation, *_ = env.step(PerModelAction(model_index=index))
    assert observation.open_unit is None
    legal = np.flatnonzero(observation.selection_mask)
    other_unit = {int(env.wargame_models[i].group_id) for i in legal}
    assert other_unit and int(env.wargame_models[0].group_id) not in other_unit


def test_remain_stationary_consumes_the_whole_unit_in_one_step() -> None:
    env = _fresh()
    before = [m.location.copy() for m in env.wargame_models]
    observation, *_ = env.step(
        PerModelAction(
            model_index=0, declaration=int(MoveDeclaration.remain_stationary)
        )
    )
    unit = int(env.wargame_models[0].group_id)
    members = [i for i, m in enumerate(env.wargame_models) if int(m.group_id) == unit]
    assert all(observation.acted_mask[i] for i in members)
    assert observation.open_unit is None
    for i in members:
        assert np.array_equal(env.wargame_models[i].location, before[i])
    assert env.model_steps == 1, "the skipped members cost no agent step"


def test_an_illegal_selection_raises() -> None:
    env = _fresh()
    env.step(PerModelAction(model_index=0))
    with pytest.raises(ValueError, match="not selectable"):
        env.step(PerModelAction(model_index=0))  # already acted
    other_unit_member = next(
        i
        for i, m in enumerate(env.wargame_models)
        if int(m.group_id) != int(env.wargame_models[0].group_id)
    )
    with pytest.raises(ValueError, match="not selectable"):
        env.step(PerModelAction(model_index=other_unit_member))  # unit lock


def test_the_selector_chooses_the_order_freely_within_the_lock() -> None:
    """Units and members may be visited in any order the lock allows."""
    env = _fresh(n_rounds=1)
    units: dict[int, list[int]] = {}
    for i, m in enumerate(env.wargame_models):
        units.setdefault(int(m.group_id), []).append(i)
    order: list[int] = []
    for unit in sorted(units, reverse=True):
        order.extend(reversed(units[unit]))
    observation = env._observe()
    terminated = False
    for index in order:
        assert not terminated
        assert observation.selection_mask[index]
        observation, _r, terminated, *_ = env.step(PerModelAction(model_index=index))
    # 1 round x 1 stepped phase, all models acted: the closing step is owed.
    assert observation.kind is StepKind.turn_close
    _obs, _r, terminated, *_ = env.step(PerModelAction())
    assert terminated


def test_one_turn_closing_step_per_round_and_the_two_clocks() -> None:
    n_rounds = 3
    env = _fresh(n_rounds)
    observation = env._observe()
    closes = 0
    model_steps = 0
    terminated = False
    while not terminated:
        if observation.kind is StepKind.turn_close:
            closes += 1
            action = PerModelAction()
        else:
            action = PerModelAction(
                model_index=int(np.flatnonzero(observation.selection_mask)[0])
            )
            model_steps += 1
        observation, _r, terminated, *_ = env.step(action)
    assert closes == n_rounds
    # `current_turn` still counts stepped phases — the old clock's meaning.
    assert env.current_turn == env.max_turns == n_rounds  # movement only
    assert env.model_steps == model_steps
    assert env.model_steps == n_rounds * len(env.wargame_models)


def test_declaration_configs_are_refused() -> None:
    config = _movement_only_config().model_copy(update={"declare_objectives": True})
    with pytest.raises(ValueError, match="declare_objectives"):
        PerModelEnv(config)


def test_the_command_phase_costs_no_agent_step() -> None:
    """With the command phase stepped, the facade traverses it without a
    decision and `current_turn` still counts it."""
    config = _movement_only_config(n_rounds=2).model_copy(
        update={
            "skip_phases": [
                BattlePhase.shooting,
                BattlePhase.charge,
                BattlePhase.pile_in,
                BattlePhase.fight,
                BattlePhase.consolidate,
            ]
        }
    )
    env = PerModelEnv(config, build_info=False)
    env.reset(seed=3)
    observation = env._observe()
    assert observation.phase is BattlePhase.movement, (
        "the first decision point is past the command phase"
    )
    assert env.current_turn == 1, "the traversed command phase still ticked"
    terminated = False
    while not terminated:
        if observation.kind is StepKind.turn_close:
            action = PerModelAction()
        else:
            action = PerModelAction(
                model_index=int(np.flatnonzero(observation.selection_mask)[0])
            )
        observation, _r, terminated, *_ = env.step(action)
    assert env.current_turn == env.max_turns == 2 * 2  # command + movement, 2 rounds
