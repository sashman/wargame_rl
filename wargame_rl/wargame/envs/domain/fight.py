"""Melee resolution: engaged units trade blows.

`docs/rules/12-fight-phase.md`. This is the fight *step* only — v1 implements no
pile-in, no overrun, no passing and no alternating activation, each recorded as
`absent` in `docs/rules/implementation-status.md` with a `DEFERRED:` tag.

What it reuses from `domain/shooting.py` is everything that is not ranged:
`resolve_attack` (the attack sequence once the skill is known),
`_allocate_target` (the defender picks which model bleeds), `DefenderStats` and
`ShootingResult`. What differs is the characteristic hit on and the absence of
cover — `12-fight-phase.md` grants none.

⚠ **Order is FIXED in v1: chargers first, then the rest, active player before
the opposing one within each.** Charging units going first is the whole of what
v1 implements of Strikes First — the rules grant the ability to a charging unit
and then alternate activation between the players, returning to the Strikes
First sub-step whenever a new Strikes First unit becomes eligible. That needs a
per-unit sequencing decision, which needs its own action space. Recorded as
`DEFERRED: fight.alternating_activation`.

⚠ **`charged_this_turn` is set only by a charge that STOOD.** A charge that ends
illegally is reverted whole and did not happen, so it earns no priority.

⚠ **A unit whose targets all died simply does not fight.** The rules would let
it *pass* and wait for an enemy pile-in to bring something into reach; with no
pile-in there is nothing to wait for. Recorded as `DEFERRED: fight.passing`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from wargame_rl.wargame.envs.domain.engagement import engagement_matrix
from wargame_rl.wargame.envs.domain.entities import WargameModel
from wargame_rl.wargame.envs.domain.pile_in import pile_in
from wargame_rl.wargame.envs.domain.shooting import (
    DefenderStats,
    ShootingResult,
    _allocate_target,
    resolve_attack,
)


@runtime_checkable
class MeleeStats(Protocol):
    """A melee weapon's stat line, structurally — the domain imports no config."""

    @property
    def attacks(self) -> int: ...
    @property
    def melee_skill(self) -> int: ...
    @property
    def strength(self) -> int: ...
    @property
    def ap(self) -> int: ...
    @property
    def damage(self) -> int: ...


@dataclass(frozen=True, slots=True)
class FightSide:
    """One force in the fight step: its models and their melee profiles.

    A record rather than a pair of positional arguments because the alternating
    scheduler indexes the two sides by seat and swaps between them every
    selection; `sides[1 - turn].weapons` is a good deal harder to get wrong than
    threading four parallel sequences through the same swap.
    """

    models: list[WargameModel]
    weapons: Sequence[Sequence[MeleeStats]]


@dataclass(frozen=True, slots=True)
class PairedFightResult:
    """One model's melee attacks against one target model.

    Deliberately NOT a reuse of `PairedShootingResult`, even though the payload
    matches. `renders/v2/scene.py::_draw_shots` draws a tracer from shooter base
    edge to target base edge for every damaging shooting result, and its "misses
    are not drawn" tuning was measured on 25-shot volleys; melee results would
    render as inch-long stubs on a mechanic that tuning never saw. Two types,
    one shared outcome record.
    """

    attacker_idx: int
    target_idx: int
    result: ShootingResult
    killed: bool = False
    target_group: int = -1


def _melee_weapon(
    weapons: Sequence[Sequence[MeleeStats]], index: int
) -> MeleeStats | None:
    """The model's melee weapon, or None if it cannot fight.

    Weapons live on the CONFIG rather than on `WargameModel`, so they are handed
    in exactly as `resolve_shooting_phase` takes `attacker_weapons` — following
    the existing seam rather than adding a second place a model's armament can
    live.

    One profile per model: `docs/rules/04-making-attacks.md` would have the
    attacker *select* one of several, which is a choice with no second case
    while every model carries at most one. `DEFERRED: fight.select_weapon`.
    """
    if index >= len(weapons):
        return None
    carried = weapons[index]
    return carried[0] if carried else None


def fight_eligible_units(
    attackers: list[WargameModel],
    defenders: list[WargameModel],
    *,
    engagement_range: float,
    base_diameter: float,
) -> dict[int, list[int]]:
    """Map each engaged attacking unit to the defending models it may strike.

    A unit fights when it is engaged. `12-fight-phase.md` also lets a unit fight
    when it *was* engaged at the start of the step — so that killing everything
    in contact does not deny you your own swing — but with no pile-in and no
    overrun nothing can become disengaged mid-step except by dying, and the
    defender allocates around casualties already.
    """
    if not attackers or not defenders:
        return {}
    matrix = engagement_matrix(
        np.array([m.location for m in attackers], dtype=float),
        np.array([m.location for m in defenders], dtype=float),
        np.array([m.is_alive for m in defenders], dtype=bool),
        np.array([m.is_alive for m in attackers], dtype=bool),
        engagement_range=engagement_range,
        base_diameter=base_diameter,
    )

    units: dict[int, list[int]] = {}
    for attacker_idx in range(len(attackers)):
        contacts = np.nonzero(matrix[attacker_idx])[0]
        if contacts.size == 0:
            continue
        units.setdefault(int(attackers[attacker_idx].group_id), []).append(attacker_idx)
    return units


def _fight_order(
    attackers: list[WargameModel], units: dict[int, list[int]]
) -> list[tuple[int, list[int]]]:
    """Charging units first, then the rest, each in group order.

    A unit counts as charging when ANY of its models does: the flag is written
    per model but the charge is a unit action, so a squad that has since lost
    the model carrying it has still charged.
    """

    def charged(members: list[int]) -> bool:
        return any(attackers[index].charged_this_turn for index in members)

    ordered = sorted(units.items())
    return [entry for entry in ordered if charged(entry[1])] + [
        entry for entry in ordered if not charged(entry[1])
    ]


def resolve_fight(
    attackers: list[WargameModel],
    defenders: list[WargameModel],
    rng: np.random.Generator,
    *,
    attacker_weapons: Sequence[Sequence[MeleeStats]],
    engagement_range: float,
    base_diameter: float,
) -> list[PairedFightResult]:
    """Every engaged attacking model strikes the unit it is in contact with.

    Units resolve one at a time — **units that charged this turn first**, then
    the rest, each in group order, and models within a unit in index order. That
    is the same determinism `resolve_shooting_phase` relies on, so a seeded
    episode reproduces.
    """
    results: list[PairedFightResult] = []
    if not attackers or not defenders:
        return results

    matrix = engagement_matrix(
        np.array([m.location for m in attackers], dtype=float),
        np.array([m.location for m in defenders], dtype=float),
        np.array([m.is_alive for m in defenders], dtype=bool),
        np.array([m.is_alive for m in attackers], dtype=bool),
        engagement_range=engagement_range,
        base_diameter=base_diameter,
    )

    units = fight_eligible_units(
        attackers,
        defenders,
        engagement_range=engagement_range,
        base_diameter=base_diameter,
    )
    for _group, member_indices in _fight_order(attackers, units):
        results.extend(
            fight_one_unit(
                attackers,
                defenders,
                member_indices,
                rng,
                matrix=matrix,
                attacker_weapons=attacker_weapons,
            )
        )
    return results


def fight_one_unit(
    attackers: list[WargameModel],
    defenders: list[WargameModel],
    member_indices: list[int],
    rng: np.random.Generator,
    *,
    matrix: np.ndarray,
    attacker_weapons: Sequence[Sequence[MeleeStats]],
) -> list[PairedFightResult]:
    """One unit's models swing, in index order.

    Split out of `resolve_fight` so the ALTERNATING scheduler can select a
    single unit at a time, which is what `12-fight-phase.md` § Fight step
    prescribes: players alternate picking one friendly eligible unit. Resolving
    a whole side at once was v1's stand-in and it is a different game -- every
    one of the active player's casualties was inflicted before any of the
    opponent's units swung back.
    """
    results: list[PairedFightResult] = []
    for attacker_idx in member_indices:
        attacker = attackers[attacker_idx]
        if not attacker.is_alive:
            continue
        weapon = _melee_weapon(attacker_weapons, attacker_idx)
        if weapon is None:
            continue
        # Re-read contact each time: a squadmate may have just killed the
        # only model this attacker was engaged with.
        contacts = [
            idx
            for idx in np.nonzero(matrix[attacker_idx])[0]
            if defenders[int(idx)].is_alive
        ]
        if not contacts:
            continue
        target_group = int(defenders[int(contacts[0])].group_id)
        members = [
            defenders[int(idx)]
            for idx in contacts
            if int(defenders[int(idx)].group_id) == target_group
        ]
        target = _allocate_target(members)
        if target is None:
            continue
        stats = DefenderStats(
            toughness=int(target.stats["toughness"]),
            save=int(target.stats["save"]),
        )
        result = resolve_attack(int(weapon.melee_skill), weapon, stats, rng)
        if result.damage_dealt:
            target.take_damage(result.damage_dealt)
        results.append(
            PairedFightResult(
                attacker_idx=attacker_idx,
                target_idx=defenders.index(target),
                result=result,
                killed=not target.is_alive,
                target_group=target_group,
            )
        )
    return results


# How far a unit must be from every enemy before its controller may PASS.
PASS_RANGE_INCHES = 5.0


@dataclass(frozen=True, slots=True)
class OverrunRules:
    """The geometry an overrun fight needs, since it makes a pile-in move first.

    Passed as a record rather than six keyword arguments because every one of
    them is already resolved through the scale at the call site, and threading
    them individually through the scheduler is how a unit conversion goes
    missing on one path and not the other.
    """

    pile_in_distance: float
    selection_range: float
    base_radius: float
    board: tuple[float, float]
    coherency_nearest: float
    coherency_furthest: float


def _contact_matrix(
    attackers: list[WargameModel],
    defenders: list[WargameModel],
    *,
    engagement_range: float,
    base_diameter: float,
) -> np.ndarray:
    """Live engagement between two forces — one definition, two callers."""
    return np.asarray(
        engagement_matrix(
            np.array([m.location for m in attackers], dtype=float),
            np.array([m.location for m in defenders], dtype=float),
            np.array([m.is_alive for m in defenders], dtype=bool),
            np.array([m.is_alive for m in attackers], dtype=bool),
            engagement_range=engagement_range,
            base_diameter=base_diameter,
        )
    )


def _strikes_first(attackers: list[WargameModel], members: list[int]) -> bool:
    """Does this unit have Strikes First? A charge is its only source here."""
    return any(attackers[index].charged_this_turn for index in members)


def _may_pass(
    attackers: list[WargameModel],
    defenders: list[WargameModel],
    units: dict[int, list[int]],
    pass_range: float,
) -> bool:
    """`12-fight-phase.md` § Passing — every eligible unit is beyond 5"."""
    if not units:
        return False
    enemies = np.array(
        [m.location for m in defenders if m.is_alive] or np.empty((0, 2)), dtype=float
    )
    if not len(enemies):
        return True
    for members in units.values():
        for index in members:
            model = attackers[index]
            if not model.is_alive:
                continue
            gap = float(
                np.linalg.norm(enemies - np.asarray(model.location), axis=1).min()
            )
            if gap <= pass_range:
                return False
    return True


def resolve_fight_step(
    sides: tuple[FightSide, FightSide],
    rng: np.random.Generator,
    *,
    engagement_range: float,
    base_diameter: float,
    pass_range: float,
    started_eligible: tuple[set[int], set[int]] | None = None,
    overrun: OverrunRules | None = None,
) -> tuple[list[PairedFightResult], list[PairedFightResult]]:
    """The fight step, with players ALTERNATING one unit at a time.

    `docs/rules/12-fight-phase.md` § Fight step, which v1 replaced with "the
    active player's whole side, then the opponent's". That stand-in is a
    materially different game: every one of the active player's casualties was
    inflicted before any of the opponent's units swung back, so a side that won
    the initiative also won every trade it started.

    The sequence, per the rules:

    1. **Strikes First combats.** Beginning with the active player, alternate
       selecting one friendly Strikes First unit that is eligible. If a player
       cannot: when *no* Strikes First unit is eligible anywhere, move to step 2
       and that player selects first there; otherwise the other player selects.
    2. **Remaining combats.** Beginning with whoever moved the sequence into
       this step, alternate selecting one friendly eligible unit.
    3. ⚠ **After any fight resolved in step 2, if a Strikes First unit has
       BECOME eligible, return to step 1.** That is reachable here because
       eligibility is recomputed against live positions and casualties.

    **Passing** (`fight.passing`, unblocked by pile-in): when the sequence
    returns to a player and every one of their eligible units is more than 5"
    from all enemies, that player may pass. Both passing in succession ends the
    step. A player with nothing eligible is not passing — they simply cannot
    select, which is the rules' other branch.

    **Overrun** (`fight.overrun`): a unit that is eligible but currently
    engaged with NOBODY -- it charged and its target died, or its contacts were
    destroyed during the step -- takes *one additional pile-in move* and then
    fights. `started_eligible` carries the units that were eligible when the
    step began, because that is the rules' own test and it cannot be recovered
    from live positions afterwards. ⚠ Expect it to fire rarely on a
    lethality-negligible blade: it needs a target to have DIED, and the shipped
    melee profile returns ~0.02 expected wounds a swing.

    `sides[0]` is the ACTIVE player. Returns each side's results in the order
    the sides were given, so the caller's attribution is unchanged.
    """
    fought: tuple[set[int], set[int]] = (set(), set())
    results: tuple[list[PairedFightResult], list[PairedFightResult]] = ([], [])
    passed = [False, False]
    turn = 0

    def eligible(seat: int) -> dict[int, list[int]]:
        """Units of `seat` that may still be selected, recomputed live."""
        side = sides[seat]
        units = fight_eligible_units(
            side.models,
            sides[1 - seat].models,
            engagement_range=engagement_range,
            base_diameter=base_diameter,
        )
        return {
            group: members
            for group, members in units.items()
            if group not in fought[seat]
        }

    # A hard bound rather than `while True`: every iteration either resolves a
    # unit, passes, or hands over, and a unit is never selected twice -- but a
    # scheduler that can hand back and forth deserves a backstop rather than a
    # proof.
    budget = 4 * (len(sides[0].models) + len(sides[1].models)) + 8
    for _ in range(budget):
        if all(passed):
            break
        available = eligible(turn)
        # ⚠ Strikes First is a SUB-STEP, not a sort key: it is re-entered
        # whenever such a unit becomes eligible, so it is tested every time
        # round rather than once at the start.
        priority = {
            group: members
            for group, members in available.items()
            if _strikes_first(sides[turn].models, members)
        }
        pool = priority or available
        if not pool and overrun is not None and started_eligible is not None:
            # ⚠ **OVERRUN.** A unit that was eligible when the step began but is
            # now engaged with nobody may still fight, by piling in first. It is
            # keyed on `started_eligible` rather than on live contact precisely
            # because the condition is "was unengaged at the start, or became
            # unengaged during the phase" -- a live read cannot tell a unit that
            # lost its target from one that never had one.
            for group in sorted(started_eligible[turn] - fought[turn]):
                members = [
                    index
                    for index, model in enumerate(sides[turn].models)
                    if model.is_alive and int(model.group_id) == group
                ]
                if not members:
                    fought[turn].add(group)
                    continue
                fought[turn].add(group)
                pile_in(
                    sides[turn].models,
                    sides[1 - turn].models,
                    eligible_units={group},
                    max_distance=overrun.pile_in_distance,
                    selection_range=overrun.selection_range,
                    engagement_range=engagement_range,
                    base_radius=overrun.base_radius,
                    board=overrun.board,
                    coherency_nearest=overrun.coherency_nearest,
                    coherency_furthest=overrun.coherency_furthest,
                )
                reached = fight_eligible_units(
                    sides[turn].models,
                    sides[1 - turn].models,
                    engagement_range=engagement_range,
                    base_diameter=base_diameter,
                )
                if group in reached:
                    matrix = _contact_matrix(
                        sides[turn].models,
                        sides[1 - turn].models,
                        engagement_range=engagement_range,
                        base_diameter=base_diameter,
                    )
                    results[turn].extend(
                        fight_one_unit(
                            sides[turn].models,
                            sides[1 - turn].models,
                            reached[group],
                            rng,
                            matrix=matrix,
                            attacker_weapons=sides[turn].weapons,
                        )
                    )
                break
            else:
                if not eligible(1 - turn):
                    break
                turn = 1 - turn
                continue
            turn = 1 - turn
            continue
        if not pool:
            if not eligible(1 - turn):
                break
            # Not a pass: this player simply cannot select, so the other does.
            turn = 1 - turn
            continue
        if _may_pass(sides[turn].models, sides[1 - turn].models, pool, pass_range):
            passed[turn] = True
            turn = 1 - turn
            continue
        passed[turn] = False
        group = min(pool)
        members = pool[group]
        fought[turn].add(group)
        matrix = _contact_matrix(
            sides[turn].models,
            sides[1 - turn].models,
            engagement_range=engagement_range,
            base_diameter=base_diameter,
        )
        results[turn].extend(
            fight_one_unit(
                sides[turn].models,
                sides[1 - turn].models,
                members,
                rng,
                matrix=matrix,
                attacker_weapons=sides[turn].weapons,
            )
        )
        turn = 1 - turn
    return results
