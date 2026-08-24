"""Shooting resolution: tabletop attack sequence (hit -> wound -> save -> damage)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from wargame_rl.wargame.envs.domain import rules_constants
from wargame_rl.wargame.envs.domain.entities import WargameModel

# Engagement range is no longer a constant here: it is authored in inches on the
# config and resolved into board units by `domain.rules_quantities`, so a
# scenario can state its own and the scale applies to it like any other rules
# distance. Read it from `BattleView.rules_quantities.engagement_range`.


@runtime_checkable
class AttackStats(Protocol):
    """The stats an attack sequence needs once its SKILL is already decided.

    Everything after the hit roll is identical for a bow and a blade, so this is
    what a melee weapon and a ranged one share. The skill itself is not here: a
    ranged weapon carries `ballistic_skill` and is modified by cover, a melee
    weapon carries `melee_skill` and is not, so it is passed in.
    """

    @property
    def attacks(self) -> int: ...
    @property
    def strength(self) -> int: ...
    @property
    def ap(self) -> int: ...
    @property
    def damage(self) -> int: ...


@runtime_checkable
class WeaponStats(AttackStats, Protocol):
    """Structural protocol for weapon stats used in resolution.

    Satisfied by ``WeaponProfile`` (Pydantic, types layer) without importing it,
    keeping the domain layer dependency-free.
    """

    @property
    def ballistic_skill(self) -> int: ...


@dataclass(frozen=True, slots=True)
class DefenderStats:
    """Target defensive stats needed for wound roll and save."""

    toughness: int
    save: int


@dataclass(frozen=True, slots=True)
class ShootingResult:
    """Outcome of one model's shooting action against one target."""

    hits: int
    wounds: int
    unsaved: int
    damage_dealt: int


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


def wound_roll_threshold(strength: int, toughness: int) -> int:
    """Return the minimum D6 roll needed to wound (2-6).

    Checks from most favourable to least. Uses integer multiplication
    to avoid rounding issues with T/2 comparison.
    """
    if 2 * toughness <= strength:
        return 2
    if strength > toughness:
        return 3
    if strength == toughness:
        return 4
    if 2 * strength <= toughness:
        return 6
    return 5


def ranged_skill(ballistic_skill: int, *, in_cover: bool = False) -> int:
    """Return the Ranged Skill an attack is resolved at.

    Cover worsens it by `COVER_RANGED_SKILL_PENALTY`
    (`docs/rules/13-terrain.md`). The returned value is the *modified* skill and
    may exceed 6 -- the unmodified-6 rule, not a clamp, is what keeps such an
    attack possible.
    """
    return ballistic_skill + (
        rules_constants.COVER_RANGED_SKILL_PENALTY if in_cover else 0
    )


def hit_probability(ballistic_skill: int, *, in_cover: bool = False) -> float:
    """Probability that one attack die hits.

    The closed form of the hit-roll table in `docs/rules/05-attack-sequence.md`:
    an unmodified 1 always fails and an unmodified 6 always hits, whatever the
    skill characteristic and whatever the modifiers say. Both bounds bite once
    cover is in play -- a Ranged Skill of 6 in cover resolves at 7, which is
    unreachable by comparison and still hits on a 6.

    Shared with :func:`resolve_shooting` through :func:`ranged_skill` so the
    dice and the analytical expectation cannot answer differently.
    """
    skill = ranged_skill(ballistic_skill, in_cover=in_cover)
    if skill > 6:
        return 1.0 / 6.0
    return (7 - max(skill, 2)) / 6.0


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


def resolve_attack(
    skill: int,
    weapon: AttackStats,
    defender: DefenderStats,
    rng: np.random.Generator,
) -> ShootingResult:
    """The attack sequence, once the skill to hit on is known.

    Shared verbatim by shooting and melee -- `docs/rules/05-attack-sequence.md`
    is one sequence, and only the characteristic it hits on and whether cover
    applies differ. Extracted rather than copied so the two can never resolve
    the same dice differently; the draws and their order are unchanged, which
    `tests/test_reward_golden.py` pins bit-for-bit.
    """
    hit_rolls = rng.integers(1, 7, size=weapon.attacks)
    hits = int(np.sum((hit_rolls != 1) & ((hit_rolls >= skill) | (hit_rolls == 6))))

    if hits == 0:
        return ShootingResult(hits=0, wounds=0, unsaved=0, damage_dealt=0)

    threshold = wound_roll_threshold(weapon.strength, defender.toughness)
    wound_rolls = rng.integers(1, 7, size=hits)
    wounds = int(
        np.sum((wound_rolls != 1) & ((wound_rolls >= threshold) | (wound_rolls == 6)))
    )

    if wounds == 0:
        return ShootingResult(hits=hits, wounds=0, unsaved=0, damage_dealt=0)

    modified_save = defender.save + weapon.ap
    save_rolls = rng.integers(1, 7, size=wounds)
    saves = int(np.sum((save_rolls != 1) & (save_rolls >= modified_save)))
    unsaved = wounds - saves

    if unsaved <= 0:
        return ShootingResult(hits=hits, wounds=wounds, unsaved=0, damage_dealt=0)

    damage_dealt = unsaved * weapon.damage
    return ShootingResult(
        hits=hits, wounds=wounds, unsaved=unsaved, damage_dealt=damage_dealt
    )


def _allocate_target(members: list[WargameModel]) -> WargameModel | None:
    """Pick which model of the target unit takes the next attack.

    The defender's choice, per the attack sequence: *"Select model. Pick a model
    in the current allocation group -- a model that has already lost Wounds if
    one is available."* Preferring the wounded model concentrates damage rather
    than spreading it, which is what stops a unit fielding a line of
    one-wound-remaining survivors.

    Allocation groups themselves are not modelled: they split a unit by
    CHARACTER and by distinct (W, Sv, InSv), and this project has one profile
    per army and no characters, so every unit is a single group. Adding the
    machinery would be a type with no second case.

    Returns None when the unit is destroyed, which is the only condition under
    which the rules discard an attack.
    """
    wounded = [m for m in members if m.is_alive and m.has_lost_wounds]
    if wounded:
        return wounded[0]
    alive = [m for m in members if m.is_alive]
    return alive[0] if alive else None


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
                target = _allocate_target(members)
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


def expected_damage(
    weapon: WeaponStats,
    defender: DefenderStats,
    *,
    in_cover: bool = False,
) -> float:
    """Closed-form analytical expected damage for one model shooting at one target.

    `in_cover` is the same flag :func:`resolve_shooting` takes and means the same
    thing: the target unit has cover against this attack, worsening its Ranged
    Skill by 1. It defaults to False, which is the expectation against a target
    in the open -- pass what the corridor actually said, or the number describes
    a shot nobody took.

    Wounds, not casualties: damage in excess of a model's remaining Wounds does
    not spill over (`docs/rules/05-attack-sequence.md`), and this does not model
    allocation, so at `damage > 1` against one-wound models it overstates what a
    volley removes from the board.
    """
    return expected_attack_damage(
        weapon.ballistic_skill, weapon, defender, in_cover=in_cover
    )


def expected_attack_damage(
    skill: int,
    weapon: AttackStats,
    defender: DefenderStats,
    *,
    in_cover: bool = False,
) -> float:
    """Closed-form expected damage once the skill to hit on is known.

    The analytical twin of :func:`resolve_attack`, and extracted for the same
    reason: a blade and a bow differ only in which characteristic they hit on
    and whether cover applies, so melee must not carry a second copy of this
    arithmetic that can drift from the dice.
    """
    p_hit = hit_probability(skill, in_cover=in_cover)
    p_wound = (7 - wound_roll_threshold(weapon.strength, defender.toughness)) / 6.0
    modified_save = defender.save + weapon.ap
    p_save = max(0.0, (7 - modified_save) / 6.0) if modified_save <= 6 else 0.0
    p_fail_save = 1.0 - p_save
    return weapon.attacks * p_hit * p_wound * p_fail_save * weapon.damage


@dataclass(frozen=True, slots=True)
class _StatBlock:
    """Row of an attacker stat array, viewed through the ``WeaponStats`` protocol."""

    attacks: int
    ballistic_skill: int
    strength: int
    ap: int
    damage: int


def expected_damage_matrix(
    attacker_stats: np.ndarray,
    defender_stats: np.ndarray,
) -> np.ndarray:
    """Expected damage for every attacker against every defender.

    Args:
        attacker_stats: ``(n_attackers, 5)`` integer array of
            ``(attacks, ballistic_skill, strength, ap, damage)``.
        defender_stats: ``(n_defenders, 2)`` integer array of
            ``(toughness, save)``.

    Returns:
        ``(n_attackers, n_defenders)`` float32 expected damage. An attacker with
        zero attacks or a defender with zero toughness scores 0 — both mean
        "no such model" rather than a real stat line, and a toughness of 0 would
        otherwise wound on 2+.

    Every entry comes from the scalar :func:`expected_damage`, so results are
    bit-identical to evaluating it per pair. The saving is that it is called once
    per *distinct* stat pair rather than once per model pair: an army built from
    one YAML profile has a single distinct pair, so a 25x25 block costs one call
    instead of 625.

    **Cover is deliberately not applied here**, so every entry is the expectation
    against a target in the open. This block is an observation input, and cover
    is a fact about a *pair of positions* rather than a pair of stat lines --
    folding it in would collapse the memoisation this function exists for (a
    distinct value per model pair, not per profile pair) and would change what
    the network sees, which is a scenario change to be measured rather than a
    correction. See `docs/shooting.md`.
    """
    n_attackers = attacker_stats.shape[0]
    n_defenders = defender_stats.shape[0]
    matrix = np.zeros((n_attackers, n_defenders), dtype=np.float32)
    if n_attackers == 0 or n_defenders == 0:
        return matrix

    unique_attackers, attacker_index = np.unique(
        attacker_stats, axis=0, return_inverse=True
    )
    unique_defenders, defender_index = np.unique(
        defender_stats, axis=0, return_inverse=True
    )

    distinct = np.zeros(
        (len(unique_attackers), len(unique_defenders)), dtype=np.float32
    )
    for i, attacker in enumerate(unique_attackers):
        if int(attacker[0]) == 0:
            continue
        weapon = _StatBlock(
            attacks=int(attacker[0]),
            ballistic_skill=int(attacker[1]),
            strength=int(attacker[2]),
            ap=int(attacker[3]),
            damage=int(attacker[4]),
        )
        for j, defender in enumerate(unique_defenders):
            if int(defender[0]) == 0:
                continue
            distinct[i, j] = expected_damage(
                weapon, DefenderStats(toughness=int(defender[0]), save=int(defender[1]))
            )

    rows = np.ravel(attacker_index)
    columns = np.ravel(defender_index)
    return distinct[rows[:, np.newaxis], columns[np.newaxis, :]]
