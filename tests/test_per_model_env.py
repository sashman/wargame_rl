"""The per-model facade's contract: the lock, the declarations, the closing step.

Every test here drives `PerModelEnv` through `step` with hand-written
decisions on the small two-unit scenario, and reads the decision point the
env hands back -- the contract is that the point states everything the policy
needs, so the tests assert on it and never on private state.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from tests.per_model_seats import random_legal_action, small_config
from wargame_rl.wargame.envs.domain.kernel.value_objects import position
from wargame_rl.wargame.envs.domain.sequencing.activation import (
    CHARGE_TARGET_DECLINE,
    MoveDeclaration,
)
from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.envs.per_model import (
    DecisionPoint,
    EpisodeOver,
    PerModelAction,
    PerModelEnv,
    StepKind,
    facade_of,
    require_per_model,
)
from wargame_rl.wargame.envs.types import WargameEnvConfig
from wargame_rl.wargame.envs.types.game_timing import BattlePhase
from wargame_rl.wargame.envs.wargame import WargameEnv


def test_the_lock_narrows_to_the_opened_unit_and_forces_the_opener() -> None:
    """Arrange the small scenario; act by opening unit 1 through model 3; assert
    the next point binds model 3, then widens to its squadmates, then to unit 0."""
    env = PerModelEnv(small_config())
    observation, info = env.reset(seed=1)
    point = observation.decision
    assert point.kind is StepKind.open and info["phase"] == "movement"
    assert point.selector_mask.tolist() == [True] * 6

    observation, reward, done, _, info = env.step(
        PerModelAction.open(3, MoveDeclaration.normal)
    )
    point = observation.decision
    assert reward == 0.0 and not done
    assert point.kind is StepKind.act
    assert point.open_unit == 1 and point.forced_model == 3
    assert np.flatnonzero(point.selector_mask).tolist() == [3]
    assert point.action_mask[3, STAY_ACTION]

    observation, *_ = env.step(PerModelAction.act(3, STAY_ACTION))
    point = observation.decision
    assert point.forced_model is None
    assert np.flatnonzero(point.selector_mask).tolist() == [4, 5]

    env.step(PerModelAction.act(4, STAY_ACTION))
    observation, *_ = env.step(PerModelAction.act(5, STAY_ACTION))
    point = observation.decision
    assert point.kind is StepKind.open and point.open_unit is None
    assert np.flatnonzero(point.selector_mask).tolist() == [0, 1, 2]
    assert point.acted[3:].all() and not point.acted[:3].any()


def test_an_illegal_decision_is_refused_and_the_point_stands() -> None:
    """A model outside the selector, a value outside the mask, and the wrong
    kind all raise with the reason; the env is unchanged afterwards."""
    env = PerModelEnv(small_config())
    observation, _ = env.reset(seed=1)
    env.step(PerModelAction.open(0, MoveDeclaration.normal))
    with pytest.raises(ValueError, match="not selectable"):
        env.step(PerModelAction.act(4, STAY_ACTION))
    with pytest.raises(ValueError, match="expected a act step"):
        env.step(PerModelAction.open(1, MoveDeclaration.normal))
    illegal = int(np.flatnonzero(~env.pending.action_mask[0])[0])  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="not legal"):
        env.step(PerModelAction.act(0, illegal))
    assert env.pending is not None and env.pending.forced_model == 0


def test_a_stationary_declaration_closes_the_unit_in_one_step() -> None:
    """Stationary marks every member acted, refreshes `previous_location`, and
    the next point is the other unit's opening."""
    env = PerModelEnv(small_config())
    observation, _ = env.reset(seed=1)
    observation, *_ = env.step(PerModelAction.open(0, MoveDeclaration.stationary))
    point = observation.decision
    assert point.kind is StepKind.open
    assert point.acted[:3].all()
    assert np.flatnonzero(point.selector_mask).tolist() == [3, 4, 5]
    for model in env.wargame_models[:3]:
        assert model.previous_location is not None
        assert np.array_equal(model.previous_location, model.location)


def test_the_movement_declaration_mask_offers_what_the_rules_allow() -> None:
    """Unengaged: stationary and normal, advance only when the scenario has it,
    never fall back."""
    env = PerModelEnv(small_config())
    observation, _ = env.reset(seed=1)
    row = observation.decision.declaration_mask[0]
    assert row.tolist() == [True, True, False, False]

    with_advance = small_config(
        skip_phases=[
            BattlePhase.charge,
            BattlePhase.pile_in,
            BattlePhase.fight,
            BattlePhase.consolidate,
        ]
    ).model_copy(update={"n_advance_speed_bins": 3})
    env = PerModelEnv(with_advance)
    observation, _ = env.reset(seed=1)
    row = observation.decision.declaration_mask[0]
    assert row.tolist() == [True, True, True, False]


def test_the_advance_roll_is_hidden_until_the_unit_declares_and_matches_the_phase_facade() -> (
    None
):
    """Before the declaration the observation shows 0; on it the unit's D6 is
    revealed on its models, and it is the roll `WargameEnv` dealt the same seed."""
    config = small_config(
        skip_phases=[
            BattlePhase.charge,
            BattlePhase.pile_in,
            BattlePhase.fight,
            BattlePhase.consolidate,
        ]
    ).model_copy(update={"n_advance_speed_bins": 3})
    old = WargameEnv(config)
    old.reset(seed=5)
    expected = [m.advance_roll for m in old.wargame_models]
    assert any(roll > 0 for roll in expected)

    env = PerModelEnv(config)
    observation, _ = env.reset(seed=5)
    assert not observation.revealed_advance_roll.any()
    observation, *_ = env.step(PerModelAction.open(3, MoveDeclaration.advance))
    revealed = observation.revealed_advance_roll
    assert revealed[3:].tolist() == expected[3:]
    assert not revealed[:3].any(), "unit 0 has not declared and its roll stays hidden"
    assert all(
        m.declared_advance and m.advanced_this_turn for m in env.wargame_models[3:]
    )
    point = observation.decision
    advance = env.player_action_handler.advance_slice
    assert advance is not None
    legal_rungs = point.action_mask[3, advance.start : advance.end]
    assert legal_rungs.any() == (expected[3] >= 2.0)


def test_a_unit_with_nothing_to_shoot_costs_no_step_and_the_turn_closes() -> None:
    """With the enemy out of range every unit is closed at the shooting phase's
    open, so after the last movement act the next decision is `close_turn`."""
    env = PerModelEnv(small_config(opponent_x=56))
    observation, _ = env.reset(seed=2)
    for unit_opener in (0, 3):
        observation, *_ = env.step(
            PerModelAction.open(unit_opener, MoveDeclaration.stationary)
        )
    point = observation.decision
    assert point.kind is StepKind.close_turn
    assert not point.selector_mask.any()


def test_the_closing_step_pays_the_window_and_opens_the_next_turn() -> None:
    """The opponent has already moved when the closing point is shown; closing
    settles one window with a reward, and the next point is round two's."""
    env = PerModelEnv(small_config())
    observation, info = env.reset(seed=2)
    before = np.array([m.location for m in env.opponent_models])
    for unit_opener in (0, 3):
        observation, *_ = env.step(
            PerModelAction.open(unit_opener, MoveDeclaration.stationary)
        )
    assert observation.decision.kind is StepKind.close_turn
    assert not np.array_equal(
        before, np.array([m.location for m in env.opponent_models])
    )
    assert observation.battle_round == 2 and observation.current_turn == 2

    observation, reward, done, _, info = env.step(PerModelAction.close_turn())
    assert info["reward_settled"] and len(info["settled"]) >= 1
    assert reward == sum(s.reward for s in info["settled"])
    assert not done and observation.decision.kind is StepKind.open
    assert info["phase"] == "movement" and info["battle_round"] == 2


@pytest.mark.parametrize(
    ("melee", "skip"),
    [
        (False, None),
        (
            False,
            [
                BattlePhase.command,
                BattlePhase.charge,
                BattlePhase.pile_in,
                BattlePhase.consolidate,
            ],
        ),
        (True, None),
        (True, [BattlePhase.fight]),
    ],
    ids=["default", "fight-stepped", "melee-all", "melee-engine-fight"],
)
def test_the_phase_clock_ends_where_the_phase_facade_does(
    melee: bool, skip: list[BattlePhase] | None
) -> None:
    """`max_turns` is the phase facade's, the episode terminates at it, and a
    step after the end raises."""
    config = small_config(melee=melee, skip_phases=skip, opponent_x=22 if melee else 56)
    env = PerModelEnv(config)
    assert env.max_turns == WargameEnv(config).max_turns
    rng = np.random.default_rng(0)
    observation, _ = env.reset(seed=4)
    done = False
    while not done:
        observation, _reward, done, _, _ = env.step(
            random_legal_action(observation.decision, rng)
        )
    assert env.current_turn == env.max_turns
    with pytest.raises(EpisodeOver):
        env.step(PerModelAction.close_turn())


def test_the_action_shape_is_validated() -> None:
    """Each kind carries exactly what it needs; the closing action carries nothing."""
    with pytest.raises(ValueError):
        PerModelAction(kind=StepKind.close_turn, model=1)
    with pytest.raises(ValueError):
        PerModelAction(kind=StepKind.open, model=-1, value=1)
    with pytest.raises(ValueError):
        PerModelAction(kind=StepKind.act, model=0, value=-1)
    assert (
        PerModelAction.target(0, CHARGE_TARGET_DECLINE).value == CHARGE_TARGET_DECLINE
    )
    assert PerModelAction.close_turn().kind is StepKind.close_turn


def test_provenance_carries_the_facade_tag_and_untagged_artefacts_read_as_phase() -> (
    None
):
    """The new facade stamps itself; the phase facade's artefacts, which carry
    no tag, are refused by the per-model loader."""
    env = PerModelEnv(small_config())
    env.reset(seed=9)
    provenance = env.provenance
    assert provenance.facade == "per_model" and provenance.seed == 9
    require_per_model(provenance)
    assert facade_of(WargameEnv(small_config()).provenance.model_dump()) == "phase"
    with pytest.raises(ValueError, match="'phase' facade"):
        require_per_model({"seed": 9, "combat_seed": 1})


def test_deferred_switches_are_refused_at_construction() -> None:
    """A scenario asking for a declaration the facade does not carry fails loudly."""
    with pytest.raises(ValueError, match="declare_objectives"):
        PerModelEnv(small_config().model_copy(update={"declare_objectives": True}))


class _WatchingSeat:
    """Wraps the opponent's adapter to read its army as its movement phase opens."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.alive_at_movement: int | None = None

    def plan(self, phase: BattlePhase, seat: Any, env: Any) -> None:
        if phase is BattlePhase.movement and self.alive_at_movement is None:
            self.alive_at_movement = sum(m.is_alive for m in seat.models)
        self.inner.plan(phase, seat, env)

    def choose(self, point: DecisionPoint, seat: Any) -> PerModelAction:
        result: PerModelAction = self.inner.choose(point, seat)
        return result


def _quiet(point: DecisionPoint) -> PerModelAction:
    """Close the turn, or open the first selectable unit with the closing declaration."""
    if point.kind is StepKind.close_turn:
        return PerModelAction.close_turn()
    return PerModelAction.open(int(np.flatnonzero(point.selector_mask)[0]), 0)


def test_attrition_culls_every_unit_on_the_board_at_each_end_of_turn() -> None:
    """`03-moving.md` § Regaining coherency: ANY unit out of coherency loses
    models at the end of each player's turn -- so the opponent's torn unit is
    culled at the end of OUR turn, before it gets a movement phase to close up.
    Arrange a torn opponent unit; act through our turn; assert the opponent
    plans its movement with the stragglers already gone."""
    base = small_config(rounds=1)
    config = WargameEnvConfig.model_validate(
        base.model_dump()
        | {"coherency": base.coherency.model_dump() | {"attrition": True}}
    )
    env = PerModelEnv(config)
    observation, _ = env.reset(seed=2)
    for model, y in zip(env.opponent_models[:3], (2.0, 18.0, 38.0), strict=True):
        model.location = position(56.0, y)
    watcher = _WatchingSeat(env.opponent_seat.adapter)
    cast(Any, env.opponent_seat).adapter = watcher
    point = observation.decision
    while point.kind is not StepKind.close_turn:
        observation, *_ = env.step(_quiet(point))
        point = observation.decision
    assert watcher.alive_at_movement == 4, "the opponent moved before it was culled"
    assert any(d.rule == "attrition.every_unit_on_the_board" for d in env.divergences)


def test_the_battle_continues_after_the_opponent_is_wiped_out() -> None:
    """`15-missions-and-scoring.md` § Ending the battle: a player with no models
    left does not lose immediately, and the survivor keeps scoring. Arrange a
    wiped opponent at the first decision; assert the clock runs to its budget,
    every round closes, and VP accrue."""
    env = PerModelEnv(small_config(rounds=3))
    observation, _ = env.reset(seed=2)
    for model in env.opponent_models:
        model.stats["current_wounds"] = 0
    closes = 0
    done = False
    while not done:
        point = observation.decision
        closes += point.kind is StepKind.close_turn
        observation, _reward, done, _, _ = env.step(_quiet(point))
    assert closes == 3 and env.current_turn == env.max_turns
    assert env.player_vp > 0
    assert any(d.rule == "battle.continues_after_a_wipe" for d in env.divergences)
