"""Shooting resolution (`10-shooting-phase.md`): the ranged skill, one shot, and a phase of declared shots."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from wargame_rl.wargame.envs.domain.attacks.allocation import allocate_target
from wargame_rl.wargame.envs.domain.attacks.sequence import ranged_skill, resolve_attack
from wargame_rl.wargame.envs.domain.attacks.stats import (
    DefenderStats,
    ShootingResult,
    WeaponStats,
)
from wargame_rl.wargame.envs.domain.kernel.entities import WargameModel


@dataclass(frozen=True, slots=True)
class PairedShootingResult:
    """A shooting result paired with attacker and target indices."""

    attacker_idx: int
    target_idx: int
    """The model the attack was **allocated** to -- the one that actually bled.

    Not the thing the attacker aimed at: the rules aim a weapon at a *unit* and
    let the defender pick which of its models takes each attack. `target_group`
    is what was declared.
    """
    result: ShootingResult
    # Whether this shot is what took the target to zero wounds. Recorded at
    # resolution time because it cannot be recovered afterwards: several
    # attackers may fire on the same target in one phase, and only one of them
    # made the kill.
    killed: bool = False
    target_group: int = -1
    """The enemy **unit** this attack was declared against. -1 when unrecorded."""
    in_cover: bool = False
    """Whether the target unit had cover against this attack.

    Recorded for the same reason as `killed`: it is a fact about the moment the
    attack resolved -- the corridor between two models that have both since
    moved -- and nothing downstream can recover it. Without it an analytical
    expectation quoted beside the result is computed under different rules than
    the dice were rolled under.
    """


def resolve_shooting(
    weapon: WeaponStats,
    defender: DefenderStats,
    rng: np.random.Generator,
    *,
    in_cover: bool = False,
) -> ShootingResult:
    """Resolve one model's shooting against one target (full attack sequence).

    Rolls D6s for hits, wounds, and saves using the provided RNG.
    Unmodified 1 always fails, unmodified 6 always succeeds.

    `in_cover` worsens the Ranged Skill by `COVER_RANGED_SKILL_PENALTY`
    (`docs/rules/13-terrain.md`): a target only partly visible is harder to hit.
    The unmodified-6 rule still applies, so cover can never make a shot
    impossible -- which is what stops it from being an absolute shield when the
    board gets crowded.
    """
    return resolve_attack(
        ranged_skill(weapon.ballistic_skill, in_cover=in_cover), weapon, defender, rng
    )


def resolve_shooting_phase(
    shots: Sequence[tuple[int, int]],
    attackers: Sequence[WargameModel],
    targets: Sequence[WargameModel],
    attacker_weapons: Sequence[Sequence[WeaponStats]],
    rng: np.random.Generator,
    cover: np.ndarray | None = None,
) -> list[PairedShootingResult]:
    """Resolve every shot declared this phase, applying damage to the targets.

    Args:
        shots: ``(attacker_idx, target_group)`` declarations already decoded from
            the action space. A weapon names an enemy **unit**, never a model --
            picking the model is the defender's job and happens here.
        attackers: Models on the firing side, indexed by ``attacker_idx``.
        targets: Models on the receiving side. Their ``group_id`` is the unit a
            declaration refers to.
        attacker_weapons: Weapon list per attacker, positionally aligned with
            *attackers*. A shorter sequence, or an empty entry, means that
            attacker cannot fire.
        rng: Dice source. Consumed only for shots that pass every check, so the
            random stream depends on how many shots resolve.
        cover: optional ``(n_attackers, n_target_groups)`` mask. Cover is a
            **unit-level** property in the rules -- a unit has it only when
            *every* model in it is in terrain or not fully visible -- so one
            model in the open denies it to the whole unit.

    Returns:
        One result per attack that resolved, in attacking-unit order.

    **Attacks are aimed at units and lost only when a unit dies.** The previous
    implementation aimed a weapon at one enemy *model* and silently discarded the
    shot when that model was already dead -- measured at a 36-40% discard rate,
    because a squad concentrating fire killed its target with the first attacks
    and threw the rest away. The rules never waste an attack that way: the
    defender allocates each one to a model still standing, and only *"excess
    attacks against a wiped-out unit are lost"*.

    **Attacking units resolve one at a time**, in group order, which is the
    rules' own sequencing -- the active player *"shoots with their units one at a
    time"*. That is also what makes deferred removal a no-op here: destruction is
    visible to the next attacking unit either way, and within one unit's
    sequence a destroyed model stops being allocatable the moment it dies.
    Removal timing would only become observable with multi-wound models, where a
    dying model still soaks allocations.

    One deliberate departure: the rules gather a unit's identical attacks into
    one pool and roll each step for the whole pool at once, working through saves
    from the lowest roll upward. This resolves attack by attack instead. With
    independent dice and a single allocation group the two are distributionally
    identical; the ordering would only bite once "already lost Wounds" can
    change mid-pool, and resolving sequentially handles that case correctly
    anyway.
    """
    results: list[PairedShootingResult] = []
    if not shots:
        return results

    groups: dict[int, list[WargameModel]] = {}
    # `id()` rather than the model: WargameModel is unhashable-by-value and
    # `list.index` would be a linear scan per allocated attack.
    index_of: dict[int, int] = {}
    for position, model in enumerate(targets):
        groups.setdefault(model.group_id, []).append(model)
        index_of[id(model)] = position

    # Attacking units fire one at a time; within a unit, its declarations are
    # gathered per target unit so a squad's attacks arrive together.
    by_attacking_unit: dict[int, dict[int, list[int]]] = {}
    for attacker_idx, target_group in shots:
        if attacker_idx >= len(attackers):
            continue
        unit = attackers[attacker_idx].group_id
        by_attacking_unit.setdefault(unit, {}).setdefault(target_group, []).append(
            attacker_idx
        )

    for unit in sorted(by_attacking_unit):
        for target_group in sorted(by_attacking_unit[unit]):
            members = groups.get(target_group)
            if not members:
                continue
            for attacker_idx in by_attacking_unit[unit][target_group]:
                attacker = attackers[attacker_idx]
                if not attacker.is_alive:
                    continue
                weapons = (
                    attacker_weapons[attacker_idx]
                    if attacker_idx < len(attacker_weapons)
                    else ()
                )
                if not weapons:
                    continue
                target = allocate_target(members)
                if target is None:
                    # The unit is wiped out. This is the one case the rules lose
                    # an attack in, and the only one this function does.
                    continue
                defender = DefenderStats(
                    toughness=target.stats["toughness"],
                    save=target.stats["save"],
                )
                in_cover = (
                    bool(cover[attacker_idx, target_group])
                    if cover is not None and target_group < cover.shape[1]
                    else False
                )
                result = resolve_shooting(weapons[0], defender, rng, in_cover=in_cover)
                killed = False
                if result.damage_dealt > 0:
                    target.take_damage(result.damage_dealt)
                    killed = not target.is_alive
                results.append(
                    PairedShootingResult(
                        attacker_idx=attacker_idx,
                        target_idx=index_of[id(target)],
                        result=result,
                        killed=killed,
                        target_group=target_group,
                        in_cover=in_cover,
                    )
                )
    return results
