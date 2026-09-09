"""The charge target step and the per-model strike: the two decisions the phase
facade never offers.

A hand-written driver charges the nearest enemy unit with every rung it can,
holds fire, and takes the first strike it is offered. Over a handful of seeds a
charge stands, the fight phase then hands the agent a strike decision whose
action mask names only the units in contact, and the blow is recorded.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.per_model_seats import charging_driver, small_config
from wargame_rl.wargame.envs.domain.kernel.value_objects import position
from wargame_rl.wargame.envs.domain.melee.consolidate import ConsolidationMode
from wargame_rl.wargame.envs.domain.sequencing.activation import (
    CHARGE_TARGET_DECLINE,
    ChargeDeclaration,
    MoveDeclaration,
    ShootDeclaration,
    ShortMoveDeclaration,
)
from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION, MoveLadder
from wargame_rl.wargame.envs.per_model import (
    DecisionPoint,
    PerModelAction,
    PerModelEnv,
    StepKind,
)
from wargame_rl.wargame.envs.per_model.phases import ShortMovePhase
from wargame_rl.wargame.envs.types.game_timing import BattlePhase


def _play_until_a_strike(seed: int) -> tuple[PerModelEnv, DecisionPoint | None, bool]:
    env = PerModelEnv(small_config(melee=True, opponent_x=22, rounds=2))
    observation, _ = env.reset(seed=seed)
    charged = False
    done = False
    while not done:
        point = observation.decision
        if (
            point.phase is BattlePhase.fight
            and point.kind is StepKind.act
            and point.seat_is_player
        ):
            return env, point, charged
        if any(m.charged_this_turn for m in env.wargame_models):
            charged = True
        observation, _reward, done, _, _ = env.step(charging_driver(env, point))
    return env, None, charged


def test_the_charge_target_step_names_units_within_reach_after_the_roll() -> None:
    """On declaring a charge the 2D6 is revealed; the target step then offers
    exactly the enemy units within 12\" whose gap the roll covers."""
    env = PerModelEnv(small_config(melee=True, opponent_x=22, rounds=2))
    observation, _ = env.reset(seed=3)
    assert not observation.revealed_charge_roll.any()
    # Stand still through movement and hold fire, to reach the charge phase.
    for opener in (0, 3):
        observation, *_ = env.step(
            PerModelAction.open(opener, MoveDeclaration.stationary)
        )
    while observation.decision.phase is BattlePhase.shooting:
        model = int(np.flatnonzero(observation.decision.selector_mask)[0])
        observation, *_ = env.step(
            PerModelAction.open(model, ShootDeclaration.hold_fire)
        )
    point = observation.decision
    assert point.phase is BattlePhase.charge and point.kind is StepKind.open
    assert point.declaration_mask[0].tolist()[:2] == [True, True]

    observation, *_ = env.step(PerModelAction.open(0, ChargeDeclaration.charge))
    point = observation.decision
    assert point.kind is StepKind.target and point.forced_model == 0
    roll = observation.revealed_charge_roll[0]
    assert roll >= 2.0 and (observation.revealed_charge_roll[:3] == roll).all()
    assert not observation.revealed_charge_roll[3:].any()
    targets = point.target_mask[0]
    quantities = env.rules_quantities
    gap = 8.0 - 2.0 * quantities.base_radius - quantities.engagement_range
    assert bool(targets[0]) == (gap <= roll)
    assert not targets[1], "the far unit is beyond 12 inches"

    observation, *_ = env.step(PerModelAction.target(0, CHARGE_TARGET_DECLINE))
    point = observation.decision
    assert point.kind is StepKind.open and point.acted[:3].all()
    assert not any(m.declared_charge for m in env.wargame_models[:3])


def test_a_standing_charge_leads_to_a_strike_the_agent_chooses() -> None:
    """Over a few seeds a charge stands; the fight then offers the agent a strike
    whose mask names only enemy units in contact, and the blow is recorded."""
    for seed in range(8):
        env, point, charged = _play_until_a_strike(seed)
        if point is None:
            continue
        assert charged
        seat = env.player_seat
        shooting = seat.handler.shooting_slice
        assert shooting is not None
        model = int(np.flatnonzero(point.selector_mask)[0])
        legal = np.flatnonzero(point.action_mask[model])
        assert legal.size >= 1
        assert all(shooting.start <= a < shooting.end for a in legal)
        assert not point.action_mask[model, STAY_ACTION]
        before = len(env.last_player_fight_results)
        env.step(PerModelAction.act(model, int(legal[0])))
        assert env.wargame_models[model].fought_this_phase
        assert len(env.last_player_fight_results) == before + 1
        return
    pytest.fail("no seed produced a standing charge in eight tries")


def _drive_to_our_consolidate(env: PerModelEnv) -> tuple[bool, set[int]]:
    """Play on from the pending point until the player decides in the consolidate
    step of ITS OWN turn; with it, the opponent units the fight step selected.

    Read from the env's own record of the fight step rather than the result
    lists, which the consolidate window clears when it opens -- and since the
    chargers all strike first, every opposing blow lands after our last one."""
    point = env.pending
    assert point is not None
    while True:
        observation, _reward, done, _, _ = env.step(charging_driver(env, point))
        point = observation.decision
        if done or point.kind is StepKind.close_turn:
            return False, set()
        if point.phase is BattlePhase.consolidate and observation.active_seat_is_player:
            return True, set(env.fought_units_of(env.opponent_seat))


def _drive_to_close_turn(env: PerModelEnv) -> bool:
    point = env.pending
    assert point is not None
    while point.kind is not StepKind.close_turn:
        observation, _reward, done, _, _ = env.step(charging_driver(env, point))
        if done:
            return False
        point = observation.decision
    return True


def test_a_charge_outlives_the_fight_step_but_not_the_fight_phase() -> None:
    """`12-fight-phase.md` makes pile-in, fight and consolidate three STEPS of one
    phase; the clock carries them as three phases. "Made a charge move this
    turn" makes a unit eligible to consolidate, so the flag must survive the
    fight step's close -- and be gone before the opponent's turn, where it would
    buy a Strikes First it did not earn. Beside it: a unit selected to fight is
    stamped fought on BOTH seats, not only where the agent struck."""
    fought_checked = False
    for seed in range(8):
        env, point, _charged = _play_until_a_strike(seed)
        if point is None:
            continue
        chargers = [i for i, m in enumerate(env.wargame_models) if m.charged_this_turn]
        assert chargers
        reached, struck = _drive_to_our_consolidate(env)
        if not reached:
            continue
        # The fight step has closed; the consolidate step of the same phase
        # still sees the charge, on every charger dead or alive.
        assert all(env.wargame_models[i].charged_this_turn for i in chargers)
        for unit in struck:
            members = [m for m in env.opponent_models if int(m.group_id) == unit]
            assert all(m.fought_this_phase for m in members)
            fought_checked = True
        assert _drive_to_close_turn(env)
        assert not any(
            m.charged_this_turn for m in (*env.wargame_models, *env.opponent_models)
        )
        if fought_checked:
            return
    pytest.fail("no seed had the opponent strike before the player consolidated")


def test_fought_expires_with_the_fight_phase_on_both_seats() -> None:
    """ "Was eligible to fight THIS phase": a unit that fought in our fight phase
    is not thereby eligible to consolidate in the opponent's. After our turn
    cycle closes, no model on either force carries the flag."""
    for seed in range(8):
        env, point, _charged = _play_until_a_strike(seed)
        if point is None:
            continue
        model = int(np.flatnonzero(point.selector_mask)[0])
        legal = np.flatnonzero(point.action_mask[model])
        env.step(PerModelAction.act(model, int(legal[0])))
        assert env.wargame_models[model].fought_this_phase
        assert _drive_to_close_turn(env)
        assert not any(
            m.fought_this_phase for m in (*env.wargame_models, *env.opponent_models)
        )
        return
    pytest.fail("no seed produced a standing charge in eight tries")


def _lone_fighter(env: PerModelEnv, at: tuple[float, float]) -> int:
    """Our unit 0 reduced to model 2, which fought this phase, standing at `at`;
    every enemy parked far away until the test places one."""
    for index in (0, 1):
        env.wargame_models[index].stats["current_wounds"] = 0
    fighter = env.wargame_models[2]
    fighter.location = position(*at)
    fighter.fought_this_phase = True
    for j, model in enumerate(env.opponent_models):
        model.location = position(50.0, 2.0 + 4.0 * j)
    return 2


def _consolidate(env: PerModelEnv) -> ShortMovePhase:
    program = ShortMovePhase(
        env, BattlePhase.consolidate, (env.player_seat, env.opponent_seat)
    )
    program.open()
    return program


def _best_short_move_toward(
    env: PerModelEnv, point: DecisionPoint, index: int, goal: tuple[float, float]
) -> int:
    handler = env.player_action_handler
    model = env.wargame_models[index]
    best, best_gap = STAY_ACTION, np.inf
    for candidate in np.flatnonzero(point.action_mask[index]):
        action = int(candidate)
        if action == STAY_ACTION:
            continue
        end = np.asarray(model.location, dtype=float) + handler.decode_action(
            action, model_idx=index, ladder=MoveLadder.short
        )
        gap = float(np.linalg.norm(np.array(goal) - end))
        if gap < best_gap:
            best, best_gap = action, gap
    return best


def _move_the_lone_fighter(
    env: PerModelEnv, program: ShortMovePhase, index: int, goal: tuple[float, float]
) -> None:
    point = program.next_decision()
    assert point is not None and point.seat_is_player and point.kind is StepKind.open
    program.apply(point, PerModelAction.open(index, ShortMoveDeclaration.move))
    point = program.next_decision()
    assert point is not None and point.kind is StepKind.act
    program.apply(
        point,
        PerModelAction.act(index, _best_short_move_toward(env, point, index, goal)),
    )
    program.next_decision()  # the unit's last member has acted: it closes here


def test_an_objective_mode_consolidation_stands_on_its_own_rule() -> None:
    """`12-fight-phase.md` § Consolidation move, Objective: a moved model ends
    within range of the objective if it can. Arrange a lone fighter inside an
    objective with every enemy far away; act with a short move that stays in
    range; assert the mode shown is Objective and the move stands."""
    env = PerModelEnv(small_config(melee=True, rounds=2))
    env.reset(seed=1)
    index = _lone_fighter(env, at=(16.0, 10.0))
    program = _consolidate(env)
    modes = program.player_consolidation_modes()
    assert modes is not None and modes[index] == ConsolidationMode.objective
    start = np.array(env.wargame_models[index].location, copy=True)
    _move_the_lone_fighter(env, program, index, goal=(16.0, 12.0))
    end = np.asarray(env.wargame_models[index].location, dtype=float)
    assert not np.array_equal(start, end), "the objective-mode move was reverted"
    offset = (
        float(np.linalg.norm(end - np.array([14.0, 10.0])))
        - env.rules_quantities.base_radius
    )
    assert offset <= 3.0


def test_an_engaging_consolidation_selects_within_three_inches_only() -> None:
    """Engaging: "the unit is within 3\" of one or more enemy units. Selection:
    one or more of those". A unit 3.24\" away is not one of those, so a move
    that engages it and walks away from the one within 3\" does not stand."""
    env = PerModelEnv(small_config(melee=True, rounds=2))
    env.reset(seed=1)
    index = _lone_fighter(env, at=(20.0, 10.0))
    near, far = env.opponent_models[0], env.opponent_models[3]
    near.location = position(20.0, 13.5)
    far.location = position(24.5, 10.0)
    diameter = 2.0 * env.rules_quantities.base_radius
    assert 3.5 - diameter <= 3.0 < 4.5 - diameter, "geometry: near within 3, far beyond"
    program = _consolidate(env)
    modes = program.player_consolidation_modes()
    assert modes is not None and modes[index] == ConsolidationMode.engaging
    start = np.array(env.wargame_models[index].location, copy=True)
    _move_the_lone_fighter(env, program, index, goal=(23.0, 10.0))
    assert np.array_equal(start, env.wargame_models[index].location), (
        "a move onto a unit beyond 3 inches stood"
    )


def test_an_engaging_consolidation_drags_the_fresh_unit_in_at_the_units_close() -> None:
    """Engaging, after moving: an enemy unit newly engaged that has not fought
    is selected by its player and strikes -- when THIS unit's move stands, not
    after both seats have consolidated."""
    env = PerModelEnv(small_config(melee=True, rounds=2))
    env.reset(seed=1)
    index = _lone_fighter(env, at=(20.0, 10.0))
    fresh = env.opponent_models[3]
    fresh.location = position(23.5, 10.0)
    program = _consolidate(env)
    modes = program.player_consolidation_modes()
    assert modes is not None and modes[index] == ConsolidationMode.engaging
    assert not env.last_opponent_fight_results
    _move_the_lone_fighter(env, program, index, goal=(23.5, 10.0))
    fighter = env.wargame_models[index]
    gap = float(
        np.linalg.norm(np.asarray(fighter.location) - np.asarray(fresh.location))
    )
    assert (
        gap - 2.0 * env.rules_quantities.base_radius
        <= env.rules_quantities.engagement_range
    )
    blows = env.last_opponent_fight_results
    assert blows and {r.attacker_idx for r in blows} == {3}
    assert int(fresh.group_id) in env.fought_units_of(env.opponent_seat)
