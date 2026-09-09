"""The attack sequence (`05-attack-sequence.md`): hit, wound, save, damage -- once the skill to hit on is known."""

from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs.domain.attacks.stats import (
    AttackStats,
    DefenderStats,
    ShootingResult,
)
from wargame_rl.wargame.envs.domain.kernel import rules_constants


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
