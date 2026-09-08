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

from tests.per_model_seats import small_config
from wargame_rl.wargame.envs.domain.activation import (
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
from wargame_rl.wargame.envs.types.game_timing import BattlePhase


def _toward_nearest_enemy(
    env: PerModelEnv, index: int, legal: np.ndarray, ladder: MoveLadder
) -> int:
    """The legal movement action that ends closest to any living enemy."""
    handler = env.player_action_handler
    model = env.wargame_models[index]
    enemies = np.array(
        [m.location for m in env.opponent_models if m.is_alive], dtype=float
    )
    best, best_gap = STAY_ACTION, np.inf
    for candidate in np.flatnonzero(legal):
        action = int(candidate)
        if action == STAY_ACTION:
            continue
        end = np.asarray(model.location, dtype=float) + handler.decode_action(
            action, model_idx=index, ladder=ladder
        )
        gap = float(np.linalg.norm(enemies - end, axis=1).min())
        if gap < best_gap:
            best, best_gap = action, gap
    return best


def _charging_driver(env: PerModelEnv, point: DecisionPoint) -> PerModelAction:
    """Close, charge the nearest unit, strike whatever is offered."""
    if point.kind is StepKind.close_turn:
        return PerModelAction.close_turn()
    model = int(np.flatnonzero(point.selector_mask)[0])
    phase = point.phase
    if point.kind is StepKind.open:
        row = point.declaration_mask[model]
        if phase is BattlePhase.movement:
            return PerModelAction.open(
                model, MoveDeclaration.normal if row[1] else MoveDeclaration.stationary
            )
        if phase is BattlePhase.shooting:
            return PerModelAction.open(model, ShootDeclaration.hold_fire)
        if phase is BattlePhase.charge:
            return PerModelAction.open(
                model, ChargeDeclaration.charge if row[1] else ChargeDeclaration.decline
            )
        return PerModelAction.open(model, ShortMoveDeclaration.decline)
    if point.kind is StepKind.target:
        targets = np.flatnonzero(point.target_mask[model])
        return PerModelAction.target(
            model, int(targets[0]) if targets.size else CHARGE_TARGET_DECLINE
        )
    legal = point.action_mask[model]
    if phase is BattlePhase.movement:
        return PerModelAction.act(
            model, _toward_nearest_enemy(env, model, legal, MoveLadder.normal)
        )
    if phase is BattlePhase.charge:
        return PerModelAction.act(
            model, _toward_nearest_enemy(env, model, legal, MoveLadder.charge)
        )
    return PerModelAction.act(model, int(np.flatnonzero(legal)[0]))


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
        observation, _reward, done, _, _ = env.step(_charging_driver(env, point))
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
    step of ITS OWN turn, collecting the opponent units that struck on the way.

    The result lists are cleared when a reward window opens, so the fight
    step's blows are read while the fight step is still the phase."""
    point = env.pending
    assert point is not None
    struck: set[int] = set()
    while True:
        observation, _reward, done, _, _ = env.step(_charging_driver(env, point))
        point = observation.decision
        struck.update(
            int(env.opponent_models[r.attacker_idx].group_id)
            for r in env.last_opponent_fight_results
        )
        if done or point.kind is StepKind.close_turn:
            return False, struck
        if point.phase is BattlePhase.consolidate and observation.active_seat_is_player:
            return True, struck


def _drive_to_close_turn(env: PerModelEnv) -> bool:
    point = env.pending
    assert point is not None
    while point.kind is not StepKind.close_turn:
        observation, _reward, done, _, _ = env.step(_charging_driver(env, point))
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
