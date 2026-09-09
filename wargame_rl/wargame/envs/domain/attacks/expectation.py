"""The analytical twin of the attack sequence: expected damage once the skill is known."""

from __future__ import annotations

from wargame_rl.wargame.envs.domain.attacks.sequence import (
    hit_probability,
    wound_roll_threshold,
)
from wargame_rl.wargame.envs.domain.attacks.stats import AttackStats, DefenderStats


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
