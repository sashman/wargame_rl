"""The fight step as a resumable sequence, so a caller can decide inside it.

`fight.py:resolve_fight_step` is the alternating-activation scheduler of
`docs/rules/12-fight-phase.md` § Fight step, written as one closed loop that
resolves both sides to the end: it picks each unit by priority and strikes with
every member at the first enemy in contact. That is right for two scripted
seats and cannot serve a seat that wants to *choose* -- which unit fights next,
and which enemy unit each striker swings at -- one decision per env step.

`FightSequence` is that loop with its control flow transcribed line for line
and its two decisions handed out as events: `Select` (this seat chooses one
unit from `pool`) and `Overrun` (the engine's own rule, resolved by the caller
through `resolve_overrun`). The caller drives it: `next()`, resolve the event,
`next()` again, until `End`. `default_choice` and `fight_one_unit` reproduce
what the closed loop would have done, so a seat with no opinion plays exactly
the old game.

`fight_one_model` is one striker's swing with the target unit chosen -- the
body of `fight_one_unit`'s loop, with `target_group` replacing "the first
contact's unit". `None` keeps the old rule.

`resolve_fight_step` itself is untouched; the whole-phase facade and its
goldens keep calling it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from wargame_rl.wargame.envs.domain.attacks.allocation import allocate_target
from wargame_rl.wargame.envs.domain.attacks.sequence import resolve_attack
from wargame_rl.wargame.envs.domain.attacks.stats import DefenderStats, MeleeStats
from wargame_rl.wargame.envs.domain.kernel.entities import WargameModel
from wargame_rl.wargame.envs.domain.melee.fight import (
    FightSide,
    OverrunRules,
    PairedFightResult,
    _contact_matrix,
    _may_pass,
    _melee_weapon,
    _strikes_first,
    fight_eligible_units,
    fight_one_unit,
)
from wargame_rl.wargame.envs.domain.melee.pile_in import pile_in


@dataclass(frozen=True, slots=True)
class Select:
    """Seat `seat` must select one unit from `pool` (group id -> striking members)."""

    seat: int
    pool: dict[int, list[int]]


@dataclass(frozen=True, slots=True)
class Overrun:
    """Seat `seat`'s unit `group` takes an overrun fight: an engine rule."""

    seat: int
    group: int


@dataclass(frozen=True, slots=True)
class End:
    """The fight step is over."""


FightEvent: TypeAlias = Select | Overrun | End


class FightSequenceError(RuntimeError):
    """The caller drove the sequence out of order."""


def default_choice(models: list[WargameModel], pool: dict[int, list[int]]) -> int:
    """The closed loop's own selection: highest priority, then lowest group id."""
    return max(
        pool,
        key=lambda candidate: (
            int(getattr(models[pool[candidate][0]], "fight_priority", 0)),
            -candidate,
        ),
    )


def fight_one_model(
    attackers: list[WargameModel],
    defenders: list[WargameModel],
    attacker_idx: int,
    rng: np.random.Generator,
    *,
    matrix: np.ndarray,
    attacker_weapons: Sequence[Sequence[MeleeStats]],
    target_group: int | None = None,
) -> PairedFightResult | None:
    """One model's melee attacks against `target_group`, or the first contact's unit.

    `matrix` is the live contact matrix (`fight._contact_matrix`) taken when
    the unit was selected; casualties since are honoured by re-reading `is_alive`,
    exactly as `fight_one_unit` does. Returns None when the model cannot swing:
    dead, unarmed, nobody in contact, or nobody left in the chosen unit.
    """
    attacker = attackers[attacker_idx]
    if not attacker.is_alive:
        return None
    weapon = _melee_weapon(attacker_weapons, attacker_idx)
    if weapon is None:
        return None
    contacts = [
        int(idx)
        for idx in np.nonzero(matrix[attacker_idx])[0]
        if defenders[int(idx)].is_alive
    ]
    if not contacts:
        return None
    if target_group is None:
        target_group = int(defenders[contacts[0]].group_id)
    members = [
        defenders[idx]
        for idx in contacts
        if int(defenders[idx].group_id) == target_group
    ]
    target = allocate_target(members)
    if target is None:
        return None
    stats = DefenderStats(
        toughness=int(target.stats["toughness"]),
        save=int(target.stats["save"]),
    )
    result = resolve_attack(int(weapon.melee_skill), weapon, stats, rng)
    if result.damage_dealt:
        target.take_damage(result.damage_dealt)
    return PairedFightResult(
        attacker_idx=attacker_idx,
        target_idx=defenders.index(target),
        result=result,
        killed=not target.is_alive,
        target_group=target_group,
    )


def contact_groups(
    attackers: list[WargameModel],
    defenders: list[WargameModel],
    attacker_idx: int,
    matrix: np.ndarray,
) -> set[int]:
    """Enemy unit ids this striker may swing at: living contacts in `matrix`."""
    return {
        int(defenders[int(idx)].group_id)
        for idx in np.nonzero(matrix[attacker_idx])[0]
        if defenders[int(idx)].is_alive
    }


class FightSequence:
    """`resolve_fight_step`'s scheduler, one event at a time.

    `sides[0]` is the ACTIVE player, as in the closed loop. `fought` records
    which units each seat has selected, and is what the consolidate step's
    drag-in clause reads afterwards.
    """

    def __init__(
        self,
        sides: tuple[FightSide, FightSide],
        *,
        engagement_range: float,
        base_diameter: float,
        pass_range: float,
        started_eligible: tuple[set[int], set[int]] | None = None,
        overrun: OverrunRules | None = None,
    ) -> None:
        self.sides = sides
        self.engagement_range = engagement_range
        self.base_diameter = base_diameter
        self.pass_range = pass_range
        self.started_eligible = started_eligible
        self.overrun = overrun
        self.fought: tuple[set[int], set[int]] = (set(), set())
        self.passed = [False, False]
        self.turn = 0
        # The closed loop's iteration budget, counted identically: every pass
        # through the loop body spends one, `continue` included.
        self._budget = 4 * (len(sides[0].models) + len(sides[1].models)) + 8
        self._spent = 0
        self._awaiting: Select | Overrun | None = None

    def eligible(self, seat: int) -> dict[int, list[int]]:
        """Units of `seat` that may still be selected, recomputed live."""
        side = self.sides[seat]
        units = fight_eligible_units(
            side.models,
            self.sides[1 - seat].models,
            engagement_range=self.engagement_range,
            base_diameter=self.base_diameter,
        )
        return {
            group: members
            for group, members in units.items()
            if group not in self.fought[seat]
        }

    def contact_matrix(self, seat: int) -> np.ndarray:
        """Live engagement between `seat`'s force and the other, for strikes."""
        return _contact_matrix(
            self.sides[seat].models,
            self.sides[1 - seat].models,
            engagement_range=self.engagement_range,
            base_diameter=self.base_diameter,
        )

    def next(self) -> FightEvent:
        """Advance to the next decision, or `End`."""
        if self._awaiting is not None:
            raise FightSequenceError(f"resolve the pending {self._awaiting} first")
        sides = self.sides
        while True:
            if self._spent >= self._budget or all(self.passed):
                return End()
            self._spent += 1
            turn = self.turn
            available = self.eligible(turn)
            priority = {
                group: members
                for group, members in available.items()
                if _strikes_first(sides[turn].models, members)
            }
            if not priority and not self.passed[1 - turn]:
                # `12-fight-phase.md` § Fight step, step 1: a player with no
                # Strikes First unit to select hands the sequence to the other
                # player while THEY still have one; only when no Strikes First
                # unit is eligible at all does the sequence move to step 2,
                # with the player who could not select going first there. A
                # seat that has just passed is not handed the sequence back.
                other_priority = {
                    group: members
                    for group, members in self.eligible(1 - turn).items()
                    if _strikes_first(sides[1 - turn].models, members)
                }
                if other_priority:
                    self.turn = 1 - turn
                    continue
            pool = priority or available
            if (
                not pool
                and self.overrun is not None
                and self.started_eligible is not None
            ):
                found = False
                for group in sorted(self.started_eligible[turn] - self.fought[turn]):
                    members = [
                        index
                        for index, model in enumerate(sides[turn].models)
                        if model.is_alive and int(model.group_id) == group
                    ]
                    if not members:
                        self.fought[turn].add(group)
                        continue
                    self.fought[turn].add(group)
                    self._awaiting = Overrun(turn, group)
                    found = True
                    break
                if found:
                    return self._awaiting  # type: ignore[return-value]
                if not self.eligible(1 - turn):
                    return End()
                self.turn = 1 - turn
                continue
            if not pool:
                if not self.eligible(1 - turn):
                    return End()
                self.turn = 1 - turn
                continue
            if _may_pass(
                sides[turn].models, sides[1 - turn].models, pool, self.pass_range
            ):
                self.passed[turn] = True
                self.turn = 1 - turn
                continue
            self.passed[turn] = False
            self._awaiting = Select(turn, pool)
            return self._awaiting

    def begin_activation(self, seat: int, group: int) -> None:
        """The selected unit is committed; strike with it, then `end_activation`."""
        pending = self._awaiting
        if not isinstance(pending, Select) or pending.seat != seat:
            raise FightSequenceError("no selection is pending for this seat")
        if group not in pending.pool:
            raise FightSequenceError(f"unit {group} is not in the selectable pool")
        self.fought[seat].add(group)

    def end_activation(self) -> None:
        """The selected unit has finished striking; the sequence hands over."""
        pending = self._awaiting
        if not isinstance(pending, Select):
            raise FightSequenceError("no activation is in progress")
        self.turn = 1 - pending.seat
        self._awaiting = None

    def resolve_selected(
        self, group: int, rng: np.random.Generator
    ) -> list[PairedFightResult]:
        """Strike with `group` exactly as the closed loop would, and hand over."""
        pending = self._awaiting
        if not isinstance(pending, Select):
            raise FightSequenceError("no selection is pending")
        seat = pending.seat
        self.begin_activation(seat, group)
        results = fight_one_unit(
            self.sides[seat].models,
            self.sides[1 - seat].models,
            pending.pool[group],
            rng,
            matrix=self.contact_matrix(seat),
            attacker_weapons=self.sides[seat].weapons,
        )
        self.end_activation()
        return results

    def resolve_overrun(self, rng: np.random.Generator) -> list[PairedFightResult]:
        """The overrun fight: one extra pile-in, then the swing; then hand over."""
        pending = self._awaiting
        if not isinstance(pending, Overrun) or self.overrun is None:
            raise FightSequenceError("no overrun is pending")
        seat, group = pending.seat, pending.group
        rules = self.overrun
        pile_in(
            self.sides[seat].models,
            self.sides[1 - seat].models,
            eligible_units={group},
            max_distance=rules.pile_in_distance,
            selection_range=rules.selection_range,
            engagement_range=self.engagement_range,
            base_radius=rules.base_radius,
            board=rules.board,
            coherency_nearest=rules.coherency_nearest,
            coherency_furthest=rules.coherency_furthest,
        )
        reached = fight_eligible_units(
            self.sides[seat].models,
            self.sides[1 - seat].models,
            engagement_range=self.engagement_range,
            base_diameter=self.base_diameter,
        )
        results: list[PairedFightResult] = []
        if group in reached:
            results = fight_one_unit(
                self.sides[seat].models,
                self.sides[1 - seat].models,
                reached[group],
                rng,
                matrix=self.contact_matrix(seat),
                attacker_weapons=self.sides[seat].weapons,
            )
        self.turn = 1 - seat
        self._awaiting = None
        return results
