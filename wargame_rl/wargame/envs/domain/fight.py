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
        engagement_range=engagement_range,
        base_diameter=base_diameter,
    )
    alive = np.array([m.is_alive for m in attackers], dtype=bool)
    matrix &= alive[:, np.newaxis]

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
