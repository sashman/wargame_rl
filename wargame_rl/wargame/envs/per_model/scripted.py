"""Seating a whole-phase script on the per-model step: the bridge adapter.

Every scripted policy in this repo answers one question -- *every model's
action for this phase* -- against the whole-phase legality mask. It never chose
an order and cannot answer a per-model decision. `ScriptedSeat` closes that
gap without touching a policy: it plans once, at the phase's open, exactly as
the phase facade would have asked, then replays the plan one decision at a
time in the order the ENGINE resolves that phase, so the board the script
lands on is the board the phase facade would have produced.

That replay order is the load-bearing part, and it is not index order:

- movement: units ascending, members in index order (the phase facade's single
  batch, since units are index-contiguous);
- shooting: units ascending, then target unit ascending, then index
  (`domain/shooting.py:resolve_shooting_phase`), non-shooters last;
- charge, pile-in, consolidate: units ascending, members in index order.

The plan is turned into per-unit declarations the script never made -- a unit
whose members all stand still is *stationary*, one with a member's advance
flag set *advanced*, one that began engaged and moves *fell back* -- so the
same rules content reaches the per-model facade's referees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

from wargame_rl.wargame.envs.domain.activation import (
    CHARGE_TARGET_DECLINE,
    ChargeDeclaration,
    MoveDeclaration,
    ShootDeclaration,
    ShortMoveDeclaration,
)
from wargame_rl.wargame.envs.domain.engagement import engaged_with_any
from wargame_rl.wargame.envs.env_components.actions import STAY_ACTION
from wargame_rl.wargame.envs.per_model.seat import Seat, full_phase_mask
from wargame_rl.wargame.envs.per_model.types import (
    DecisionPoint,
    PerModelAction,
    StepKind,
)
from wargame_rl.wargame.envs.types import WargameEnvAction
from wargame_rl.wargame.envs.types.game_timing import BattlePhase

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.domain.entities import WargameModel
    from wargame_rl.wargame.envs.per_model.env import PerModelEnv


class PhasePolicy(Protocol):
    """What both policy hierarchies share: one action per model, per phase."""

    def select_action(
        self,
        models: list[WargameModel],
        env: Any,
        action_mask: np.ndarray | None = None,
    ) -> WargameEnvAction: ...


class ScriptedSeatError(RuntimeError):
    """The plan and the program disagree about what is legal -- a bug, not play."""


@dataclass
class _UnitPlan:
    group: int
    declaration: int
    order: list[int]
    actions: dict[int, int] = field(default_factory=dict)
    target: int | None = None


class ScriptedSeat:
    """Plan once per phase, replay in engine order."""

    def __init__(self, policy: PhasePolicy, *, shoots: bool) -> None:
        self.policy = policy
        self.shoots = shoots
        self._plans: list[_UnitPlan] = []
        self._phase: BattlePhase | None = None

    def plan(self, phase: BattlePhase, seat: Seat, env: PerModelEnv) -> None:
        """Ask the script exactly once for this phase, against its usual mask."""
        mask = full_phase_mask(env, seat, phase, shooting_overlay=self.shoots)
        plan = self.policy.select_action(seat.models, cast(Any, env), action_mask=mask)
        self._phase = phase
        self._plans = []
        if phase is BattlePhase.command:
            seat.handler.declare_move_types(plan, seat.models)
            return
        if phase is BattlePhase.fight:
            seat.handler.declare_fight_order(plan, seat.models)
            return
        if phase is BattlePhase.movement:
            self._plans = _movement_plans(plan, seat, env)
        elif phase is BattlePhase.shooting:
            self._plans = _shooting_plans(plan, seat)
        elif phase is BattlePhase.charge:
            self._plans = _charge_plans(plan, seat)
        elif phase in (BattlePhase.pile_in, BattlePhase.consolidate):
            self._plans = _short_move_plans(plan, seat, phase)

    def choose(self, point: DecisionPoint, seat: Seat) -> PerModelAction:
        """The next decision of the plan that `point` admits."""
        if point.kind is StepKind.open:
            return self._choose_open(point, seat)
        if point.kind is StepKind.target:
            return self._choose_target(point, seat)
        if point.kind is StepKind.act:
            return self._choose_act(point, seat)
        raise ScriptedSeatError("a scripted seat never closes the turn")

    def _plan_for(self, group: int) -> _UnitPlan | None:
        for plan in self._plans:
            if plan.group == group:
                return plan
        return None

    def _choose_open(self, point: DecisionPoint, seat: Seat) -> PerModelAction:
        selector = point.selector_mask
        for plan in self._plans:
            for index in plan.order:
                if selector[index]:
                    if not point.declaration_mask[index, plan.declaration]:
                        raise ScriptedSeatError(
                            f"unit {plan.group} planned declaration "
                            f"{plan.declaration} which the phase does not allow"
                        )
                    return PerModelAction.open(index, plan.declaration)
        # A unit the program admits that the plan never mentioned: it takes the
        # closing declaration, which every phase makes legal.
        index = int(np.flatnonzero(selector)[0])
        return PerModelAction.open(index, 0)

    def _choose_target(self, point: DecisionPoint, seat: Seat) -> PerModelAction:
        index = int(np.flatnonzero(point.selector_mask)[0])
        plan = self._plan_for(int(seat.models[index].group_id))
        target = None if plan is None else plan.target
        if target is None or not point.target_mask[index, target]:
            return PerModelAction.target(index, CHARGE_TARGET_DECLINE)
        return PerModelAction.target(index, target)

    def _choose_act(self, point: DecisionPoint, seat: Seat) -> PerModelAction:
        selector = point.selector_mask
        open_unit = point.open_unit
        plan = None if open_unit is None else self._plan_for(open_unit)
        order = plan.order if plan is not None else list(np.flatnonzero(selector))
        candidates = [i for i in order if selector[i]] or list(np.flatnonzero(selector))
        index = int(candidates[0])
        action = STAY_ACTION if plan is None else plan.actions.get(index, STAY_ACTION)
        if not point.action_mask[index, action]:
            # The plan's target died, or its rung is no longer reachable: the
            # phase facade would have resolved nothing for it either.
            action = STAY_ACTION
        return PerModelAction.act(index, action)


def _members(seat: Seat, group: int) -> list[int]:
    return seat.unit_members(group)


def _movement_plans(
    plan: WargameEnvAction, seat: Seat, env: PerModelEnv
) -> list[_UnitPlan]:
    engaged = np.zeros(seat.n_models, dtype=bool)
    alive_enemies = seat.alive_enemies()
    if env.config.melee.enabled and alive_enemies:
        engaged = engaged_with_any(
            np.array([m.location for m in seat.models], dtype=float),
            np.array([m.location for m in alive_enemies], dtype=float),
            np.ones(len(alive_enemies), dtype=bool),
            seat.alive(),
            engagement_range=env.rules_quantities.engagement_range,
            base_diameter=2.0 * env.rules_quantities.base_radius,
        )
    plans: list[_UnitPlan] = []
    for group in seat.living_units():
        members = _members(seat, group)
        actions = {i: int(plan.actions[i]) for i in members}
        moves = any(a != STAY_ACTION for a in actions.values())
        if seat.models[members[0]].declared_advance:
            declaration = int(MoveDeclaration.advance)
        elif bool(engaged[members].any()) and moves:
            declaration = int(MoveDeclaration.fall_back)
        elif not moves:
            declaration = int(MoveDeclaration.stationary)
        else:
            declaration = int(MoveDeclaration.normal)
        plans.append(_UnitPlan(group, declaration, members, actions))
    return plans


def _shooting_plans(plan: WargameEnvAction, seat: Seat) -> list[_UnitPlan]:
    shooting = seat.handler.shooting_slice
    plans: list[_UnitPlan] = []
    for group in seat.living_units():
        members = _members(seat, group)
        shooters: list[tuple[int, int]] = []
        actions: dict[int, int] = {}
        for index in members:
            action = int(plan.actions[index])
            if shooting is not None and shooting.start <= action < shooting.end:
                shooters.append((action - shooting.start, index))
                actions[index] = action
            else:
                actions[index] = STAY_ACTION
        shooters.sort()
        order = [index for _target, index in shooters] + [
            index for index in members if actions[index] == STAY_ACTION
        ]
        declaration = int(
            ShootDeclaration.shoot if shooters else ShootDeclaration.hold_fire
        )
        plans.append(_UnitPlan(group, declaration, order, actions))
    return plans


def _nearest_enemy_unit(seat: Seat, members: list[int]) -> int | None:
    best: tuple[float, int] | None = None
    for index in members:
        here = np.asarray(seat.models[index].location, dtype=float)
        for enemy in seat.enemies:
            if not enemy.is_alive:
                continue
            gap = float(np.linalg.norm(np.asarray(enemy.location, dtype=float) - here))
            if best is None or gap < best[0]:
                best = (gap, int(enemy.group_id))
    return None if best is None else best[1]


def _charge_plans(plan: WargameEnvAction, seat: Seat) -> list[_UnitPlan]:
    plans: list[_UnitPlan] = []
    for group in seat.living_units():
        members = _members(seat, group)
        actions = {i: int(plan.actions[i]) for i in members}
        moves = any(a != STAY_ACTION for a in actions.values())
        declared = bool(seat.models[members[0]].declared_charge)
        if declared and moves:
            plans.append(
                _UnitPlan(
                    group,
                    int(ChargeDeclaration.charge),
                    members,
                    actions,
                    target=_nearest_enemy_unit(seat, members),
                )
            )
        else:
            plans.append(_UnitPlan(group, int(ChargeDeclaration.decline), members, {}))
    return plans


def _short_move_plans(
    plan: WargameEnvAction, seat: Seat, phase: BattlePhase
) -> list[_UnitPlan]:
    plans: list[_UnitPlan] = []
    for group in seat.living_units():
        members = _members(seat, group)
        actions = {i: int(plan.actions[i]) for i in members}
        moves = any(a != STAY_ACTION for a in actions.values())
        declaration = int(
            ShortMoveDeclaration.move if moves else ShortMoveDeclaration.decline
        )
        plans.append(_UnitPlan(group, declaration, members, actions if moves else {}))
    return plans
