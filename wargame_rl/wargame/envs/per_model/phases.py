"""Phase programs: what one decision does in each phase, and when a unit closes.

A program owns one phase of the clock for the seats that act in it. It states
the next legal decision (`next_decision`), applies one (`apply`), and runs the
unit-close referees. It holds no rule of its own: every legality question goes
to the action handler's helpers or the domain, every move to `resolve_move`
and `back_off_to_unengaged`, every shot to `resolve_shooting_phase`, every
strike to `fight_one_model`, every referee to `domain/unit_referees.py`.

The per-model move body is `ActionHandler.apply`'s inner loop, one model at a
time, with the friendly bases re-read live so an earlier mover's ground is
taken -- the whole-phase facade resolves movement in exactly this order, which
is what lets a script replayed through here land on the same board.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import numpy as np

from wargame_rl.wargame.envs.domain.kernel.dice import DicePurpose
from wargame_rl.wargame.envs.domain.kernel.value_objects import position
from wargame_rl.wargame.envs.domain.melee.charge import charge_stands
from wargame_rl.wargame.envs.domain.melee.consolidate import (
    ConsolidationMode,
    consolidation_mode,
    objective_consolidation_stands,
    objective_in_reach,
)
from wargame_rl.wargame.envs.domain.melee.fight import (
    PASS_RANGE_INCHES,
    FightSide,
    OverrunRules,
    fight_eligible_units,
)
from wargame_rl.wargame.envs.domain.melee.fight_sequence import (
    End,
    FightEvent,
    FightSequence,
    Overrun,
    Select,
    contact_groups,
    default_choice,
    fight_one_model,
)
from wargame_rl.wargame.envs.domain.melee.pile_in import (
    SELECTION_RANGE_INCHES,
    short_move_stands,
)
from wargame_rl.wargame.envs.domain.movement.coherency_enforcement import (
    coherency_after_unit_move,
)
from wargame_rl.wargame.envs.domain.movement.engagement import engaged_with_any
from wargame_rl.wargame.envs.domain.movement.fall_back import fall_back_stands
from wargame_rl.wargame.envs.domain.movement.moves import (
    back_off_to_unengaged,
    resolve_move,
)
from wargame_rl.wargame.envs.domain.movement.unit_moves import (
    revert_unit,
    touched_enemy_units,
    unit_moved,
)
from wargame_rl.wargame.envs.domain.sequencing.activation import (
    CHARGE_TARGET_DECLINE,
    ChargeDeclaration,
    MoveDeclaration,
    PhaseActivation,
    ShootDeclaration,
    ShortMoveDeclaration,
)
from wargame_rl.wargame.envs.domain.shooting.cover import unit_cover_mask
from wargame_rl.wargame.envs.domain.shooting.resolve import resolve_shooting_phase
from wargame_rl.wargame.envs.env_components.actions import (
    STAY_ACTION,
    MoveLadder,
    _base_arrays,
)
from wargame_rl.wargame.envs.env_components.distance_cache import compute_distances
from wargame_rl.wargame.envs.per_model.seat import Seat, unit_shooting_masks
from wargame_rl.wargame.envs.per_model.types import (
    N_DECLARATIONS,
    DecisionPoint,
    PerModelAction,
    StepKind,
)
from wargame_rl.wargame.envs.types.game_timing import BattlePhase

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.per_model.env import PerModelEnv


def move_model(
    env: PerModelEnv,
    seat: Seat,
    index: int,
    action: int,
    *,
    ladder: MoveLadder,
    engagement_rings: bool,
) -> None:
    """Resolve one model's move exactly as `ActionHandler.apply` does per model.

    Bases: enemies block, friendlies may be crossed but not ended on, and the
    friendly positions are read live. The engagement rings are the movement
    phase's end-state rule; the charge, pile-in and consolidate drop them
    because ending engaged is their whole point.
    """
    quantities = env.rules_quantities
    radius = quantities.base_radius
    model = seat.models[index]
    lower = position(radius, radius)
    upper = position(env.board_width - radius, env.board_height - radius)
    collides = radius > 0.0
    blocker_centres, blocker_radii = _base_arrays(seat.enemies if collides else None)
    alive_enemies = seat.alive_enemies()
    if engagement_rings and quantities.engagement_range > 0.0 and alive_enemies:
        engagement_centres = np.array([m.location for m in alive_enemies], dtype=float)
        engagement_reach = np.array(
            [
                quantities.engagement_range + float(m.base_radius) + radius
                for m in alive_enemies
            ],
            dtype=float,
        )
    else:
        engagement_centres = np.empty((0, 2), dtype=float)
        engagement_reach = np.empty(0, dtype=float)
    model.previous_location = model.location.copy()
    displacement = seat.handler.decode_action(
        action, model_idx=index, advance_roll=model.advance_roll, ladder=ladder
    )
    if not collides:
        model.location = back_off_to_unengaged(
            model.location,
            np.clip(model.location + displacement, lower, upper),
            engagement_centres,
            engagement_reach,
        )
        return
    friendly = [
        other for j, other in enumerate(seat.models) if j != index and other.is_alive
    ]
    if friendly:
        friendly_centres = np.array([m.location for m in friendly], dtype=float)
        friendly_radii = np.array([m.base_radius for m in friendly], dtype=float)
    else:
        friendly_centres = np.zeros((0, 2), dtype=float)
        friendly_radii = np.zeros(0, dtype=float)
    in_bounds = np.clip(model.location + displacement, lower, upper)
    occupied_centres = np.concatenate([blocker_centres, friendly_centres])
    occupied_reach = np.concatenate([blocker_radii, friendly_radii]) + model.base_radius
    model.location = back_off_to_unengaged(
        model.location,
        resolve_move(
            model.location,
            in_bounds - model.location,
            model.base_radius,
            blocker_centres,
            blocker_radii,
            friendly_centres,
            friendly_radii,
        ),
        engagement_centres,
        engagement_reach,
        occupied_centres,
        occupied_reach,
    )


def _engaged_units(seat: Seat, env: PerModelEnv) -> set[int]:
    """Group ids of `seat` with any living member in engagement range."""
    quantities = env.rules_quantities
    return {
        int(seat.models[i].group_id)
        for i in range(seat.n_models)
        if seat.models[i].is_alive
    } & _engaged_groups(seat, env, quantities.engagement_range)


def _engaged_groups(seat: Seat, env: PerModelEnv, engagement_range: float) -> set[int]:
    alive_enemies = seat.alive_enemies()
    if not alive_enemies or engagement_range <= 0.0:
        return set()
    engaged = engaged_with_any(
        np.array([m.location for m in seat.models], dtype=float),
        np.array([m.location for m in alive_enemies], dtype=float),
        np.ones(len(alive_enemies), dtype=bool),
        seat.alive(),
        engagement_range=engagement_range,
        base_diameter=2.0 * env.rules_quantities.base_radius,
    )
    return {int(seat.models[i].group_id) for i in np.flatnonzero(engaged)}


class PhaseProgram:
    """One phase of the clock, for the seats that act in it."""

    def __init__(self, env: PerModelEnv, phase: BattlePhase) -> None:
        self.env = env
        self.phase = phase

    def open(self) -> None:
        """Consult scripted seats and settle which units are in the phase."""

    def next_decision(self) -> DecisionPoint | None:
        """The next decision anyone must take, or None once the phase is done."""
        return None

    def apply(self, point: DecisionPoint, action: PerModelAction) -> None:
        """Resolve one decision."""

    def close(self) -> None:
        """The phase is over: carry what the boundary hook needs."""

    def player_consolidation_modes(self) -> np.ndarray | None:
        """Per-model compulsory consolidation mode, when this is that phase."""
        return None

    def acted_mask(self) -> np.ndarray | None:
        """Who has acted in the activation a pending decision belongs to.

        Read by the env around `apply`, so a step's effect can name the models
        it marked acted -- a skip declaration marks a whole unit with no member
        taking a step. None when no activation is open.
        """
        return None


class _UnitPhase(PhaseProgram):
    """A single-seat phase with a declaration on every unit's opening step."""

    def __init__(self, env: PerModelEnv, phase: BattlePhase, seat: Seat) -> None:
        super().__init__(env, phase)
        self.seat = seat
        self.activation = PhaseActivation(seat.n_models)
        self.units: set[int] = set()
        self.awaiting_target = False

    def acted_mask(self) -> np.ndarray | None:
        return self.activation.acted

    # --- what a subclass states -------------------------------------------

    def declaration_mask(self, unit: int) -> np.ndarray:
        raise NotImplementedError

    def auto_closes(self, unit: int) -> bool:
        """True when the unit's only legal declaration is the closing one."""
        mask = self.declaration_mask(unit)
        return not bool(mask[1:].any())

    def on_open(self, unit: int, declaration: int) -> None:
        raise NotImplementedError

    def on_target(self, unit: int, target: int) -> None:
        raise NotImplementedError

    def target_mask(self, unit: int) -> np.ndarray:
        return np.zeros(self.seat.n_enemy_units(), dtype=bool)

    def action_mask(self, index: int) -> np.ndarray:
        raise NotImplementedError

    def on_act(self, index: int, action: int) -> None:
        raise NotImplementedError

    def on_unit_close(self, unit: int, members: list[int]) -> None:
        raise NotImplementedError

    # --- the program ----------------------------------------------------------

    def open(self) -> None:
        if self.seat.adapter is not None:
            self.seat.adapter.plan(self.phase, self.seat, self.env)
        self.units = set(self.seat.living_units())
        for unit in sorted(self.units):
            if self.auto_closes(unit):
                self.activation.close_without_steps(unit, self.seat.unit_members(unit))

    def _members_left(self) -> np.ndarray:
        return self.activation.selectable(
            self.seat.alive(), self.seat.group_ids, self.units
        )

    def next_decision(self) -> DecisionPoint | None:
        activation = self.activation
        while True:
            if activation.is_open:
                selectable = self._members_left()
                if not selectable.any():
                    self._close_open_unit()
                    continue
                if self.awaiting_target:
                    return self._point(
                        StepKind.target,
                        selectable,
                        target_mask=self._rows(
                            selectable,
                            lambda i: self.target_mask(int(activation.open_unit or 0)),
                            self.seat.n_enemy_units(),
                        ),
                    )
                return self._point(
                    StepKind.act,
                    selectable,
                    action_mask=self._rows(
                        selectable, self.action_mask, self.seat.handler.n_actions
                    ),
                )
            selectable = self._members_left()
            if not selectable.any():
                return None
            return self._point(
                StepKind.open,
                selectable,
                declaration_mask=self._rows(
                    selectable,
                    lambda i: self.declaration_mask(int(self.seat.models[i].group_id)),
                    N_DECLARATIONS,
                ),
            )

    def apply(self, point: DecisionPoint, action: PerModelAction) -> None:
        index = action.model
        group = int(self.seat.models[index].group_id)
        if action.kind is StepKind.open:
            self.activation.open(group, index, action.value, force_opener=True)
            self.on_open(group, action.value)
            return
        if action.kind is StepKind.target:
            self.awaiting_target = False
            self.on_target(group, action.value)
            return
        self.on_act(index, action.value)
        self.activation.mark_acted(index, group)

    def _close_open_unit(self) -> None:
        unit = int(self.activation.open_unit or 0)
        members = self.seat.unit_members(unit)
        self.on_unit_close(unit, members)
        self.awaiting_target = False
        self.activation.close()

    def close_now(self) -> None:
        """Close the open unit from its declaration: no member takes a step."""
        unit = int(self.activation.open_unit or 0)
        for index in self.seat.unit_members(unit):
            self.activation.acted[index] = True
        self.awaiting_target = False
        self.activation.close()

    def capture_start(self, unit: int) -> None:
        self.activation.start_positions = {
            index: np.array(self.seat.models[index].location, copy=True)
            for index in self.seat.unit_members(unit, alive_only=False)
        }

    def _rows(
        self,
        selectable: np.ndarray,
        row_for: Callable[[int], np.ndarray],
        width: int,
    ) -> np.ndarray:
        rows = np.zeros((self.seat.n_models, width), dtype=bool)
        for index in np.flatnonzero(selectable):
            row = row_for(int(index))
            rows[index, : row.shape[0]] = row
        return rows

    def _point(
        self,
        kind: StepKind,
        selector: np.ndarray,
        *,
        declaration_mask: np.ndarray | None = None,
        action_mask: np.ndarray | None = None,
        target_mask: np.ndarray | None = None,
    ) -> DecisionPoint:
        n = self.seat.n_models
        return DecisionPoint(
            kind=kind,
            seat_is_player=self.seat.is_player,
            phase=self.phase,
            selector_mask=selector,
            declaration_mask=(
                declaration_mask
                if declaration_mask is not None
                else np.zeros((n, N_DECLARATIONS), dtype=bool)
            ),
            action_mask=(
                action_mask
                if action_mask is not None
                else np.zeros((n, self.seat.handler.n_actions), dtype=bool)
            ),
            target_mask=(
                target_mask
                if target_mask is not None
                else np.zeros((n, self.seat.n_enemy_units()), dtype=bool)
            ),
            acted=self.activation.acted.copy(),
            open_unit=self.activation.open_unit,
            forced_model=self.activation.forced_model,
        )


class MovementPhase(_UnitPhase):
    """`09-movement-phase.md`: declare the move type, then move model by model."""

    def __init__(self, env: PerModelEnv, seat: Seat) -> None:
        super().__init__(env, BattlePhase.movement, seat)
        self.engaged_units: set[int] = set()

    def open(self) -> None:
        if self.env.config.melee.enabled:
            self.engaged_units = _engaged_units(self.seat, self.env)
        super().open()

    def declaration_mask(self, unit: int) -> np.ndarray:
        engaged = unit in self.engaged_units
        return np.array(
            [
                True,
                not engaged,
                not engaged and self.seat.handler.advance_slice is not None,
                engaged,
            ],
            dtype=bool,
        )

    def auto_closes(self, unit: int) -> bool:
        return False

    def on_open(self, unit: int, declaration: int) -> None:
        members = self.seat.unit_members(unit, alive_only=False)
        self.capture_start(unit)
        if declaration == MoveDeclaration.stationary:
            for index in members:
                model = self.seat.models[index]
                if model.is_alive:
                    model.previous_location = model.location.copy()
            self.close_now()
            return
        if declaration == MoveDeclaration.advance:
            for index in members:
                self.seat.models[index].declared_advance = True
                self.seat.models[index].advanced_this_turn = True
            self.env.reveal_advance_roll(self.seat, unit)

    def on_target(self, unit: int, target: int) -> None:
        raise NotImplementedError("the movement phase has no target step")

    def action_mask(self, index: int) -> np.ndarray:
        handler = self.seat.handler
        mask: np.ndarray = handler.registry.get_action_mask(self.phase).copy()
        advance = handler.advance_slice
        if advance is not None:
            legality = handler.advance_legality(self.seat.models, self.seat.enemies)
            mask[advance.start : advance.end] &= legality[index]
        return mask

    def on_act(self, index: int, action: int) -> None:
        move_model(
            self.env,
            self.seat,
            index,
            action,
            ladder=MoveLadder.normal,
            engagement_rings=True,
        )

    def on_unit_close(self, unit: int, members: list[int]) -> None:
        env = self.env
        quantities = env.rules_quantities
        starts = self.activation.start_positions
        nearest = quantities.scale.to_units(env.config.coherency.nearest_distance)
        furthest = quantities.scale.to_units(env.config.coherency.furthest_distance)
        if self.activation.declaration == MoveDeclaration.fall_back and members:
            if unit_moved(self.seat.models, members, starts):
                if fall_back_stands(
                    self.seat.models,
                    members,
                    self.seat.alive_enemies(),
                    engagement_range=quantities.engagement_range,
                    base_diameter=2.0 * quantities.base_radius,
                    coherency_nearest=nearest,
                    coherency_furthest=furthest,
                ):
                    for index in self.seat.unit_members(unit, alive_only=False):
                        self.seat.models[index].fell_back_this_turn = True
                else:
                    revert_unit(self.seat.models, members, starts)
        coherency_after_unit_move(
            self.seat.models, members, nearest, furthest, env.coherency_mode
        )


class ShootingPhase(_UnitPhase):
    """`10-shooting-phase.md`: a unit declares it shoots; each model names a unit;
    the unit's attacks resolve together when it closes.

    Three orderings from `04-making-attacks.md` and `05-attack-sequence.md`
    meet here. Every target is selected before any attack is resolved, so a
    member names its unit without seeing a squadmate's dice -- the shots are
    only declared on the `act` steps and resolved at the unit's close. An
    attack at a unit a squadmate's dice have just wiped is therefore LOST, as
    `05` § 4 says, not re-aimed. And destroyed models are removed "only after
    the attacking unit has resolved all of its attacks", so the NEXT unit
    selects its targets, and has its cover judged, against the board as it
    then stands: the legality masks and the membership behind "every model in
    cover" are rebuilt whenever a close has removed a model.

    Within the unit the shots resolve in target-unit order, then model index --
    the phase facade's order for one action vector -- so a script through both
    facades draws the same dice for the same shot.
    """

    def __init__(self, env: PerModelEnv, seat: Seat) -> None:
        super().__init__(env, BattlePhase.shooting, seat)
        self.unit_masks = np.zeros((seat.n_models, 0), dtype=bool)
        self.members: dict[int, list[int]] = {}
        self.declared: list[tuple[int, int]] = []
        self._alive_key: np.ndarray = np.zeros(0, dtype=bool)
        self._phase_open_masks = self.unit_masks
        self._phase_open_members: dict[int, list[int]] = {}

    def open(self) -> None:
        self._refresh(force=True)
        self._phase_open_masks = self.unit_masks
        self._phase_open_members = self.members
        super().open()

    def _refresh(self, *, force: bool = False) -> None:
        """Rebuild what the board's casualties change, only when they did."""
        key = np.array([m.is_alive for m in self.seat.enemies], dtype=bool)
        if not force and np.array_equal(key, self._alive_key):
            return
        self._alive_key = key
        self.unit_masks = unit_shooting_masks(self.env, self.seat)
        self.members = {}
        for j, enemy in enumerate(self.seat.enemies):
            if enemy.is_alive:
                self.members.setdefault(int(enemy.group_id), []).append(j)
        if force:
            return
        # The phase facade judged every unit's targets at phase open; where the
        # casualties just removed change what a unit still to act may target,
        # the two games part here.
        pending = np.flatnonzero(~self.activation.acted & self.seat.alive())
        if pending.size and not np.array_equal(
            self.unit_masks[pending], self._phase_open_masks[pending]
        ):
            self.env.note_divergence("shooting.targets_judged_after_casualties")

    def _unit_may_shoot(self, unit: int) -> bool:
        rows = self.unit_masks[self.seat.unit_members(unit)]
        return bool(rows.size and rows.any())

    def declaration_mask(self, unit: int) -> np.ndarray:
        mask = np.zeros(N_DECLARATIONS, dtype=bool)
        mask[ShootDeclaration.hold_fire] = True
        mask[ShootDeclaration.shoot] = self._unit_may_shoot(unit)
        return mask

    def on_open(self, unit: int, declaration: int) -> None:
        self.declared = []
        if declaration == ShootDeclaration.hold_fire:
            self.close_now()

    def on_target(self, unit: int, target: int) -> None:
        raise NotImplementedError("the shooting phase has no target step")

    def action_mask(self, index: int) -> np.ndarray:
        handler = self.seat.handler
        mask = np.zeros(handler.n_actions, dtype=bool)
        mask[STAY_ACTION] = True
        shooting = handler.shooting_slice
        if shooting is not None and self.unit_masks.shape[1]:
            mask[shooting.start : shooting.end] = self.unit_masks[index]
        return mask

    def on_act(self, index: int, action: int) -> None:
        if action == STAY_ACTION:
            return
        shooting = self.seat.handler.shooting_slice
        if shooting is None:
            return
        self.declared.append((index, action - shooting.start))

    def on_unit_close(self, unit: int, members: list[int]) -> None:
        shots = sorted(self.declared, key=lambda shot: (shot[1], shot[0]))
        self.declared = []
        if not shots:
            return
        width = self.unit_masks.shape[1]
        empty = np.zeros((self.seat.n_models, width), dtype=bool)
        # Cover for every shot of the unit, judged once before any resolves,
        # against the members this unit faces; a squadmate's kill during the
        # resolution below changes nothing, as `05` § Suffering damage says.
        cover = unit_cover_mask(
            self.env.visibility_between, shots, self.seat.models, self.seat.enemies
        )
        if cover is None:
            cover = empty
        at_open = np.zeros(len(self.seat.enemies), dtype=bool)
        for members in self._phase_open_members.values():
            at_open[members] = True
        if not np.array_equal(at_open, self._alive_key):
            cover_at_open = unit_cover_mask(
                self.env.visibility_between,
                shots,
                self.seat.models,
                self.seat.enemies,
                alive=at_open,
            )
            if cover_at_open is None:
                cover_at_open = empty
            if any(cover[i, g] != cover_at_open[i, g] for i, g in shots):
                self.env.note_divergence("shooting.cover_judged_after_casualties")
        for index, group in shots:
            # One call per shot keeps the dice tagged with the model rolling
            # them; `resolve_shooting_phase` loses the shot itself when the
            # target unit is already wiped, which is the rule.
            self.seat.shooting_results.extend(
                resolve_shooting_phase(
                    shots=[(index, group)],
                    attackers=self.seat.models,
                    targets=self.seat.enemies,
                    attacker_weapons=self.seat.ranged_weapons,
                    rng=self.env.roller(
                        DicePurpose.shooting, self.seat, unit=unit, model=index
                    ),
                    cover=cover,
                )
            )
        self._refresh()


class ChargePhase(_UnitPhase):
    """`11-charge-phase.md`: declare, roll, name the target, then move."""

    def __init__(self, env: PerModelEnv, seat: Seat) -> None:
        super().__init__(env, BattlePhase.charge, seat)
        self.eligible: set[int] = set()

    def open(self) -> None:
        if self.env.config.melee.enabled:
            self.eligible = self.seat.handler.charge_eligible_units(
                self.seat.models, self.seat.enemies
            )
        super().open()

    def declaration_mask(self, unit: int) -> np.ndarray:
        mask = np.zeros(N_DECLARATIONS, dtype=bool)
        mask[ChargeDeclaration.decline] = True
        mask[ChargeDeclaration.charge] = unit in self.eligible
        return mask

    def on_open(self, unit: int, declaration: int) -> None:
        members = self.seat.unit_members(unit, alive_only=False)
        if declaration == ChargeDeclaration.decline:
            for index in members:
                self.seat.models[index].declared_charge = False
            self.close_now()
            return
        for index in members:
            self.seat.models[index].declared_charge = True
        self.env.reveal_charge_roll(self.seat, unit)
        self.awaiting_target = True

    def _reach(self, unit: int) -> float:
        members = self.seat.unit_members(unit)
        if not members:
            return 0.0
        roll = float(self.seat.models[members[0]].charge_roll)
        if roll <= 0.0:
            return 0.0
        return float(self.env.rules_quantities.scale.to_units(roll))

    def target_mask(self, unit: int) -> np.ndarray:
        """Enemy units within 12" of the unit AND within the roll's reach."""
        quantities = self.env.rules_quantities
        width = self.seat.n_enemy_units()
        mask = np.zeros(width, dtype=bool)
        members = self.seat.unit_members(unit)
        alive_enemies = self.seat.alive_enemies()
        if not members or not alive_enemies:
            return mask
        positions = np.array(
            [self.seat.models[i].location for i in members], dtype=float
        )
        enemy_positions = np.array([m.location for m in alive_enemies], dtype=float)
        enemy_radii = np.array(
            [float(m.base_radius) for m in alive_enemies], dtype=float
        )
        centre_gaps = np.linalg.norm(
            positions[:, np.newaxis, :] - enemy_positions[np.newaxis, :, :], axis=2
        )
        within = (
            centre_gaps - 2.0 * quantities.base_radius
            <= quantities.scale.to_units(self.env.config.melee.charge_range)
        )
        gaps = centre_gaps - enemy_radii[np.newaxis, :] - quantities.base_radius
        reach = self._reach(unit)
        for j, enemy in enumerate(alive_enemies):
            group = int(enemy.group_id)
            if group >= width or mask[group]:
                continue
            column_within = bool(within[:, j].any())
            column_reach = bool(
                (gaps[:, j].min() - quantities.engagement_range) <= reach
            )
            if column_within and column_reach:
                mask[group] = True
        return mask

    def on_target(self, unit: int, target: int) -> None:
        members = self.seat.unit_members(unit, alive_only=False)
        if target == CHARGE_TARGET_DECLINE:
            for index in members:
                self.seat.models[index].declared_charge = False
            self.close_now()
            return
        self.activation.set_charge_target(target)
        self.capture_start(unit)

    def action_mask(self, index: int) -> np.ndarray:
        handler = self.seat.handler
        mask = np.zeros(handler.n_actions, dtype=bool)
        mask[STAY_ACTION] = True
        movement = handler.movement_slice
        legality = handler.charge_legality(self.seat.models, self.seat.enemies)
        mask[movement.start : movement.end] = legality[index]
        return mask

    def on_act(self, index: int, action: int) -> None:
        move_model(
            self.env,
            self.seat,
            index,
            action,
            ladder=MoveLadder.charge,
            engagement_rings=False,
        )

    def on_unit_close(self, unit: int, members: list[int]) -> None:
        starts = self.activation.start_positions
        target = self.activation.charge_target
        if (
            not members
            or target is None
            or not unit_moved(self.seat.models, members, starts)
        ):
            return
        quantities = self.env.rules_quantities
        stands = charge_stands(
            self.seat.models,
            members,
            self.seat.alive_enemies(),
            starts,
            target_group=target,
            reach=self._reach(unit),
            engagement_range=quantities.engagement_range,
            base_diameter=2.0 * quantities.base_radius,
            coherency_nearest=quantities.scale.to_units(
                self.env.config.coherency.nearest_distance
            ),
            coherency_furthest=quantities.scale.to_units(
                self.env.config.coherency.furthest_distance
            ),
        )
        if stands:
            for index in members:
                self.seat.models[index].charged_this_turn = True
        else:
            revert_unit(self.seat.models, members, starts)


class _ShortMoveSeat(_UnitPhase):
    """One seat's pile-in or consolidate: declare, then move up to 3"."""

    def __init__(self, env: PerModelEnv, phase: BattlePhase, seat: Seat) -> None:
        super().__init__(env, phase, seat)
        self.modes: dict[int, ConsolidationMode] = {}
        self.touched_before: dict[int, set[int]] = {}
        self.offsets_before: np.ndarray | None = None

    def open(self) -> None:
        if self.seat.adapter is not None:
            self.seat.adapter.plan(self.phase, self.seat, self.env)
        self.units = set()
        if not self.env.config.melee.enabled:
            return
        legality = self.seat.handler.short_move_legality(
            self.seat.models, self.seat.enemies, self.phase
        )
        for unit in self.seat.living_units():
            rows = legality[self.seat.unit_members(unit)]
            if not (rows.size and rows.any()):
                continue
            if self.phase is BattlePhase.consolidate:
                mode = self._mode(unit)
                if mode is ConsolidationMode.none:
                    continue
                self.modes[unit] = mode
            self.units.add(unit)

    def _offsets(self) -> np.ndarray | None:
        env = self.env
        if not env.objectives:
            return None
        return compute_distances(
            self.seat.models, env.objectives
        ).model_obj_norms_offset

    def _consolidate_distance(self) -> float:
        return self.env.rules_quantities.scale.to_units(
            self.env.config.melee.consolidate_distance
        )

    def _touched(self, unit: int) -> set[int]:
        quantities = self.env.rules_quantities
        return touched_enemy_units(
            self.seat.models,
            self.seat.unit_members(unit),
            self.seat.alive_enemies(),
            engagement_range=quantities.engagement_range,
            base_diameter=2.0 * quantities.base_radius,
        )

    def _mode(self, unit: int) -> ConsolidationMode:
        quantities = self.env.rules_quantities
        return consolidation_mode(
            self.seat.models,
            self.seat.unit_members(unit),
            self.seat.alive_enemies(),
            self._offsets(),
            engagement_range=quantities.engagement_range,
            base_diameter=2.0 * quantities.base_radius,
            consolidate_distance=self._consolidate_distance(),
        )

    def declaration_mask(self, unit: int) -> np.ndarray:
        mask = np.zeros(N_DECLARATIONS, dtype=bool)
        mask[ShortMoveDeclaration.decline] = True
        mask[ShortMoveDeclaration.move] = unit in self.units
        return mask

    def on_open(self, unit: int, declaration: int) -> None:
        if declaration == ShortMoveDeclaration.decline:
            self.close_now()
            return
        self.capture_start(unit)
        if self.phase is BattlePhase.consolidate:
            self.touched_before[unit] = self._touched(unit)
            self.offsets_before = self._offsets()

    def on_target(self, unit: int, target: int) -> None:
        raise NotImplementedError("a short move has no target step")

    def action_mask(self, index: int) -> np.ndarray:
        handler = self.seat.handler
        mask = np.zeros(handler.n_actions, dtype=bool)
        mask[STAY_ACTION] = True
        movement = handler.movement_slice
        legality = handler.short_move_legality(
            self.seat.models, self.seat.enemies, self.phase
        )
        mask[movement.start : movement.end] = legality[index]
        return mask

    def on_act(self, index: int, action: int) -> None:
        move_model(
            self.env,
            self.seat,
            index,
            action,
            ladder=MoveLadder.short,
            engagement_rings=False,
        )

    def on_unit_close(self, unit: int, members: list[int]) -> None:
        starts = self.activation.start_positions
        if not members or not unit_moved(self.seat.models, members, starts):
            return
        if self.phase is BattlePhase.pile_in:
            stands = self._short_move_stands(
                members,
                starts,
                self.env.rules_quantities.scale.to_units(SELECTION_RANGE_INCHES),
            )
        else:
            stands = self._consolidation_stands(unit, members, starts)
        if not stands:
            revert_unit(self.seat.models, members, starts)
            return
        if (
            self.phase is BattlePhase.consolidate
            and self.modes.get(unit) is ConsolidationMode.engaging
        ):
            # Engaging, after moving: every enemy unit this move newly engaged
            # that has not been selected to fight this phase is selected now,
            # by the opposing player, and strikes -- at this unit's close, not
            # after both seats have finished, so the order is the chapter's.
            dragged = self._touched(unit) - self.touched_before.get(unit, set())
            self.env.drag_in(self.seat, dragged)

    def _short_move_stands(
        self, members: list[int], starts: dict[int, np.ndarray], selection_range: float
    ) -> bool:
        quantities = self.env.rules_quantities
        return short_move_stands(
            self.seat.models,
            members,
            self.seat.alive_enemies(),
            starts,
            selection_range=selection_range,
            engagement_range=quantities.engagement_range,
            base_radius=quantities.base_radius,
            coherency_nearest=quantities.scale.to_units(
                self.env.config.coherency.nearest_distance
            ),
            coherency_furthest=quantities.scale.to_units(
                self.env.config.coherency.furthest_distance
            ),
        )

    def _consolidation_stands(
        self, unit: int, members: list[int], starts: dict[int, np.ndarray]
    ) -> bool:
        """`12-fight-phase.md` § Consolidation move: each mode has its own test.

        Ongoing is a pile-in onto the units already engaged (selection range
        zero); Engaging selects among units within 3", not the pile-in's 5";
        Objective is judged by the engine's own objective rule. One referee for
        all three was the pile-in's, under which a unit in Objective mode with
        a living enemy anywhere was reverted every time it moved.
        """
        mode = self.modes.get(unit, ConsolidationMode.none)
        if mode is ConsolidationMode.ongoing:
            return self._short_move_stands(members, starts, 0.0)
        if mode is ConsolidationMode.engaging:
            return self._short_move_stands(
                members, starts, self._consolidate_distance()
            )
        if mode is ConsolidationMode.objective:
            offsets_after = self._offsets()
            before = self.offsets_before
            if offsets_after is None or before is None:
                return False
            objective = objective_in_reach(
                before, members, self._consolidate_distance()
            )
            if objective is None:
                return False
            quantities = self.env.rules_quantities
            return objective_consolidation_stands(
                self.seat.models,
                members,
                starts,
                offsets_before=before,
                offsets_after=offsets_after,
                objective=objective,
                radius=float(self.env.objectives[objective].radius_size),
                coherency_nearest=quantities.scale.to_units(
                    self.env.config.coherency.nearest_distance
                ),
                coherency_furthest=quantities.scale.to_units(
                    self.env.config.coherency.furthest_distance
                ),
            )
        return False


class ShortMovePhase(PhaseProgram):
    """`12-fight-phase.md` pile-in / consolidate: both players, active first."""

    def __init__(
        self, env: PerModelEnv, phase: BattlePhase, seats: tuple[Seat, Seat]
    ) -> None:
        super().__init__(env, phase)
        self.seats = seats
        self.programs: list[_ShortMoveSeat] = []
        self.current = 0

    def open(self) -> None:
        self.programs = [
            _ShortMoveSeat(self.env, self.phase, seat) for seat in self.seats
        ]
        self.current = 0
        self.programs[0].open()

    def next_decision(self) -> DecisionPoint | None:
        while self.current < len(self.programs):
            point = self.programs[self.current].next_decision()
            if point is not None:
                return point
            self.current += 1
            if self.current < len(self.programs):
                self.programs[self.current].open()
        return None

    def apply(self, point: DecisionPoint, action: PerModelAction) -> None:
        self.programs[self.current].apply(point, action)

    def acted_mask(self) -> np.ndarray | None:
        if self.current >= len(self.programs):
            return None
        return self.programs[self.current].acted_mask()

    def player_consolidation_modes(self) -> np.ndarray | None:
        if self.phase is not BattlePhase.consolidate:
            return None
        for program in self.programs:
            if program.seat.is_player:
                modes = np.zeros(program.seat.n_models, dtype=np.int64)
                for index, model in enumerate(program.seat.models):
                    modes[index] = int(
                        program.modes.get(int(model.group_id), ConsolidationMode.none)
                    )
                return modes
        return None


class FightPhase(PhaseProgram):
    """`12-fight-phase.md` § Fight step, with a seat choosing its strikes."""

    def __init__(self, env: PerModelEnv, seats: tuple[Seat, Seat]) -> None:
        super().__init__(env, BattlePhase.fight)
        self.seats = seats
        self.sequence: FightSequence | None = None
        self.event: FightEvent | None = None
        self.activation: PhaseActivation | None = None
        self.matrix: np.ndarray | None = None
        self.pool: dict[int, list[int]] = {}

    def acted_mask(self) -> np.ndarray | None:
        return None if self.activation is None else self.activation.acted

    def open(self) -> None:
        env = self.env
        if not env.config.melee.enabled:
            return
        for seat in self.seats:
            if seat.adapter is not None:
                seat.adapter.plan(self.phase, seat, env)
        quantities = env.rules_quantities
        engagement_range = quantities.engagement_range
        base_diameter = 2.0 * quantities.base_radius
        started = tuple(
            set(
                fight_eligible_units(
                    seat.models,
                    seat.enemies,
                    engagement_range=engagement_range,
                    base_diameter=base_diameter,
                )
            )
            for seat in self.seats
        )
        overrun = (
            OverrunRules(
                pile_in_distance=quantities.scale.to_units(
                    env.config.melee.pile_in_distance
                ),
                selection_range=quantities.scale.to_units(SELECTION_RANGE_INCHES),
                base_radius=quantities.base_radius,
                board=(float(env.board_width), float(env.board_height)),
                coherency_nearest=quantities.scale.to_units(
                    env.config.coherency.nearest_distance
                ),
                coherency_furthest=quantities.scale.to_units(
                    env.config.coherency.furthest_distance
                ),
            )
            if env.config.melee.overrun
            else None
        )
        self.sequence = FightSequence(
            (
                FightSide(
                    models=self.seats[0].models, weapons=self.seats[0].melee_weapons
                ),
                FightSide(
                    models=self.seats[1].models, weapons=self.seats[1].melee_weapons
                ),
            ),
            engagement_range=engagement_range,
            base_diameter=base_diameter,
            pass_range=quantities.scale.to_units(PASS_RANGE_INCHES),
            started_eligible=(started[0], started[1]),
            overrun=overrun,
        )

    def _rng(
        self, seat_index: int, unit: int | None, model: int | None
    ) -> np.random.Generator:
        return self.env.roller(
            DicePurpose.melee, self.seats[seat_index], unit=unit, model=model
        )

    def next_decision(self) -> DecisionPoint | None:
        sequence = self.sequence
        if sequence is None:
            return None
        while True:
            if self.event is None:
                self.event = sequence.next()
            event = self.event
            if isinstance(event, End):
                return None
            if isinstance(event, Overrun):
                results = sequence.resolve_overrun(
                    self._rng(event.seat, event.group, None)
                )
                self.seats[event.seat].fight_results.extend(results)
                self.event = None
                continue
            event = cast(Select, event)
            seat = self.seats[event.seat]
            if seat.adapter is not None:
                group = default_choice(seat.models, event.pool)
                results = sequence.resolve_selected(
                    group, self._rng(event.seat, group, None)
                )
                seat.fight_results.extend(results)
                self.event = None
                continue
            point = self._agent_point(event, seat)
            if point is None:
                continue
            return point

    def _agent_point(self, event: Select, seat: Seat) -> DecisionPoint | None:
        sequence = self.sequence
        assert sequence is not None
        if self.activation is None:
            self.activation = PhaseActivation(seat.n_models)
            self.pool = event.pool
        activation = self.activation
        alive = seat.alive()
        if activation.is_open:
            unit = int(activation.open_unit or 0)
            assert self.matrix is not None
            selectable = np.zeros(seat.n_models, dtype=bool)
            for index in self.pool.get(unit, []):
                if (
                    alive[index]
                    and not activation.acted[index]
                    and contact_groups(seat.models, seat.enemies, index, self.matrix)
                ):
                    selectable[index] = True
            if not selectable.any():
                sequence.end_activation()
                activation.close()
                self.activation = None
                self.matrix = None
                self.event = None
                return None
        else:
            selectable = np.zeros(seat.n_models, dtype=bool)
            for members in self.pool.values():
                for index in members:
                    if alive[index] and not activation.acted[index]:
                        selectable[index] = True
            if not selectable.any():
                return None
        rows = np.zeros((seat.n_models, seat.handler.n_actions), dtype=bool)
        shooting = seat.handler.shooting_slice
        matrix = (
            self.matrix
            if self.matrix is not None
            else sequence.contact_matrix(event.seat)
        )
        for row in np.flatnonzero(selectable):
            groups = contact_groups(seat.models, seat.enemies, int(row), matrix)
            if shooting is None:
                continue
            for group in groups:
                if group < shooting.size:
                    rows[row, shooting.start + group] = True
        return DecisionPoint(
            kind=StepKind.act,
            seat_is_player=seat.is_player,
            phase=self.phase,
            selector_mask=selectable,
            declaration_mask=np.zeros((seat.n_models, N_DECLARATIONS), dtype=bool),
            action_mask=rows,
            target_mask=np.zeros((seat.n_models, seat.n_enemy_units()), dtype=bool),
            acted=activation.acted.copy(),
            open_unit=activation.open_unit,
            forced_model=None,
        )

    def apply(self, point: DecisionPoint, action: PerModelAction) -> None:
        sequence = self.sequence
        event = self.event
        activation = self.activation
        assert (
            sequence is not None
            and isinstance(event, Select)
            and activation is not None
        )
        seat = self.seats[event.seat]
        index = action.model
        group = int(seat.models[index].group_id)
        if not activation.is_open:
            sequence.begin_activation(event.seat, group)
            activation.open(group, index, None, force_opener=False)
            self.matrix = sequence.contact_matrix(event.seat)
            for member in seat.unit_members(group, alive_only=False):
                seat.models[member].fought_this_phase = True
        shooting = seat.handler.shooting_slice
        target = action.value - shooting.start if shooting is not None else None
        assert self.matrix is not None
        result = fight_one_model(
            seat.models,
            seat.enemies,
            index,
            self._rng(event.seat, group, index),
            matrix=self.matrix,
            attacker_weapons=seat.melee_weapons,
            target_group=target,
        )
        if result is not None:
            seat.fight_results.append(result)
        activation.mark_acted(index, group)

    def close(self) -> None:
        env = self.env
        if self.sequence is None:
            return
        env.carry_fight_to_consolidate(
            self.seats,
            (set(self.sequence.fought[0]), set(self.sequence.fought[1])),
        )


__all__ = [
    "ChargePhase",
    "FightPhase",
    "MovementPhase",
    "PhaseProgram",
    "ShootingPhase",
    "ShortMovePhase",
    "move_model",
    "touched_enemy_units",
]
