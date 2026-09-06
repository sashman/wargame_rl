"""Seats a whole-phase scripted policy in the per-model facade.

A ``BaselinePolicy`` returns every model's action in one call and never chose
an order. The adapter plans **once per phase, at its first step**, and replays
the plan one model at a time — so the scripted bar means the same thing in
both facades, which is what makes it transferable across the architecture
race (issue #283's comparability contract).

**Replay order is the whole-phase facade's RESOLUTION order, not blindly the
model index order.** Movement (and every other displacing phase) resolves by
model index, so canonical order reproduces it; shooting resolves in
(attacking unit ascending, declared target unit ascending, declaration order)
— ``domain.shooting.resolve_shooting_phase`` sorts that way — so the adapter
replays shots in exactly that order, or the dice stream would shift whenever a
unit's members split their fire across target units out of order.

Declarations: the old facade declares move types in the command phase; the
per-model facade re-times them to the unit's opening step. Scripts *plan* from
a state in which every declaration is already made (their movement plan reads
``declared_advance``), so the adapter derives the per-unit declarations from
the script's own command plan and applies them en bloc through
``declare_for_phase`` before planning — game-equivalent, since a declaration
only affects its own unit's members.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.baseline.policy import BaselinePolicy
from wargame_rl.wargame.envs.env_components.actions import (
    MOVE_TYPE_ADVANCE,
    MOVE_TYPE_CHARGE,
    STAY_ACTION,
)
from wargame_rl.wargame.envs.per_model.types import (
    ChargeDeclaration,
    MoveDeclaration,
    PerModelAction,
    PerModelObservation,
    ShootDeclaration,
    StepKind,
)
from wargame_rl.wargame.envs.types import BattlePhase, WargameEnvAction

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.per_model.facade import PerModelEnv


class ScriptedPolicyAdapter:
    """Drives a ``PerModelEnv`` with a whole-phase ``BaselinePolicy``.

    One adapter per episode: it caches its phase plan by ``current_turn``,
    which restarts at zero on reset.
    """

    def __init__(self, policy: BaselinePolicy) -> None:
        self._policy = policy
        self._plan_key: int | None = None
        self._queue: list[PerModelAction] = []
        self._command_key: int | None = None
        self._advancing: set[int] = set()
        self._charging: set[int] = set()

    def next_action(
        self, env: PerModelEnv, observation: PerModelObservation
    ) -> PerModelAction:
        """The next step to submit for the observation the env just returned."""
        if observation.kind is StepKind.turn_close:
            return PerModelAction(model_index=None)
        if self._plan_key != env.current_turn:
            self._plan_phase(env)
            self._plan_key = env.current_turn
        if not self._queue:
            raise RuntimeError(
                "The adapter's phase plan is exhausted but the env still owes "
                "a model step — the plan and the facade disagree about who "
                "acts this phase."
            )
        return self._queue.pop(0)

    # -- Planning -------------------------------------------------------------

    def _plan_phase(self, env: PerModelEnv) -> None:
        phase = env.game_clock_state.phase or BattlePhase.movement
        self._ensure_command_plan(env, phase)

        # Declarations first: the plan and the mask must both be computed from
        # a state in which every declaration is made, exactly as the old
        # facade's command step precedes its movement observation.
        if phase is BattlePhase.movement and self._advancing:
            env.declare_for_phase(
                {unit: int(MoveDeclaration.advance) for unit in self._advancing}
            )
        if phase is BattlePhase.charge and self._charging:
            env.declare_for_phase(
                {unit: int(ChargeDeclaration.charge) for unit in self._charging}
            )

        plan = self._policy.select_action(
            env.wargame_models, env, action_mask=env.current_action_mask()
        )
        if phase is BattlePhase.shooting:
            self._queue = self._shooting_queue(env, plan)
        elif phase is BattlePhase.fight:
            self._queue = self._fight_queue(env, plan)
        elif phase is BattlePhase.charge:
            self._queue = self._charge_queue(env, plan)
        else:
            self._queue = self._canonical_queue(env, plan)

    def _ensure_command_plan(self, env: PerModelEnv, phase: BattlePhase) -> None:
        """Derive the turn's move-type declarations from the script's command plan.

        Only where the old facade stepped the command phase at all — otherwise
        the script never declared anything and every unit moves normally.
        """
        if phase not in (BattlePhase.movement, BattlePhase.charge):
            return
        state = env.game_clock_state
        key = state.battle_round or 0
        if self._command_key == key:
            return
        self._command_key = key
        self._advancing = set()
        self._charging = set()
        if BattlePhase.command in set(env.config.skip_phases):
            return
        handler = env.player_action_handler
        advance_action = handler.move_type_action(MOVE_TYPE_ADVANCE)
        charge_action = handler.move_type_action(MOVE_TYPE_CHARGE)
        if advance_action is None and charge_action is None:
            return
        command_plan = self._policy.select_command(env.wargame_models, env)
        for unit, leader in _unit_leaders(env).items():
            if leader >= len(command_plan.actions):
                continue
            chosen = int(command_plan.actions[leader])
            if advance_action is not None and chosen == advance_action:
                self._advancing.add(unit)
            elif charge_action is not None and chosen == charge_action:
                self._charging.add(unit)

    # -- Queues ---------------------------------------------------------------

    def _canonical_queue(
        self, env: PerModelEnv, plan: WargameEnvAction
    ) -> list[PerModelAction]:
        """Movement, pile-in, consolidate: replay in model index order."""
        queue: list[PerModelAction] = []
        for unit, members in _units_in_order(env):
            for index in members:
                queue.append(
                    PerModelAction(model_index=index, action=int(plan.actions[index]))
                )
        return queue

    def _shooting_queue(
        self, env: PerModelEnv, plan: WargameEnvAction
    ) -> list[PerModelAction]:
        """Shooting: replay in the engine's own resolution order.

        Within a unit, shots resolve sorted by declared target unit and then by
        declaration (= index) order; members that hold fire draw no dice and
        follow. A unit with no shot at all is consumed by a single
        ``hold_fire`` opener.
        """
        shooting_slice = env.player_action_handler.shooting_slice
        queue: list[PerModelAction] = []
        for unit, members in _units_in_order(env):
            shooters: list[tuple[int, int]] = []
            holders: list[int] = []
            for index in members:
                act = int(plan.actions[index])
                if (
                    shooting_slice is not None
                    and shooting_slice.start <= act < shooting_slice.end
                ):
                    shooters.append((act - shooting_slice.start, index))
                else:
                    holders.append(index)
            if not shooters:
                queue.append(
                    PerModelAction(
                        model_index=members[0],
                        action=STAY_ACTION,
                        declaration=int(ShootDeclaration.hold_fire),
                    )
                )
                continue
            ordered = [index for _target, index in sorted(shooters)] + holders
            for index in ordered:
                queue.append(
                    PerModelAction(model_index=index, action=int(plan.actions[index]))
                )
        return queue

    def _charge_queue(
        self, env: PerModelEnv, plan: WargameEnvAction
    ) -> list[PerModelAction]:
        """Charge: declared units replay their moves; the rest decline."""
        queue: list[PerModelAction] = []
        for unit, members in _units_in_order(env):
            if unit in self._charging:
                for index in members:
                    queue.append(
                        PerModelAction(
                            model_index=index, action=int(plan.actions[index])
                        )
                    )
            else:
                queue.append(
                    PerModelAction(
                        model_index=members[0],
                        action=STAY_ACTION,
                        declaration=int(ChargeDeclaration.decline),
                    )
                )
        return queue

    def _fight_queue(
        self, env: PerModelEnv, plan: WargameEnvAction
    ) -> list[PerModelAction]:
        """Fight: one opening step per unit carrying its activation priority."""
        fight_slice = env.player_action_handler.fight_order_slice
        queue: list[PerModelAction] = []
        for unit, members in _units_in_order(env):
            leader = members[0]
            priority = 0
            if fight_slice is not None and leader < len(plan.actions):
                chosen = int(plan.actions[leader])
                if fight_slice.start <= chosen < fight_slice.end:
                    priority = chosen - fight_slice.start
            queue.append(
                PerModelAction(
                    model_index=leader, action=STAY_ACTION, declaration=priority
                )
            )
        return queue


def _unit_leaders(env: PerModelEnv) -> dict[int, int]:
    """Each unit's leader: its lowest-indexed alive model (the engine's rule)."""
    leaders: dict[int, int] = {}
    for index, model in enumerate(env.wargame_models):
        if not model.is_alive:
            continue
        leaders.setdefault(int(model.group_id), index)
    return leaders


def _units_in_order(env: PerModelEnv) -> list[tuple[int, list[int]]]:
    """Units in ascending id order, each with its alive members ascending.

    Group ids are assigned in index-contiguous blocks, so visiting units in id
    order visits models in exactly the sequence a whole-force index-order batch
    did — which is what the bridge's bit-identity rests on.
    """
    units: dict[int, list[int]] = {}
    for index, model in enumerate(env.wargame_models):
        if not model.is_alive:
            continue
        units.setdefault(int(model.group_id), []).append(index)
    return sorted(units.items())
