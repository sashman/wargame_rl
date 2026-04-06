"""Shooting resolution: tabletop attack sequence (hit -> wound -> save -> damage)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ENGAGEMENT_RANGE = 1
"""Grid-cell distance within which models are considered engaged (v3.0 stub)."""


@dataclass(frozen=True, slots=True)
class ShootingResult:
    """Outcome of one model's shooting action against one target."""

    hits: int
    wounds: int
    unsaved: int
    damage_dealt: int


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


def resolve_shooting(
    attacks: int,
    ballistic_skill: int,
    strength: int,
    ap: int,
    damage: int,
    target_toughness: int,
    target_save: int,
    rng: np.random.Generator,
) -> ShootingResult:
    """Resolve one model's shooting against one target (full attack sequence).

    Rolls D6s for hits, wounds, and saves using the provided RNG.
    Unmodified 1 always fails, unmodified 6 always succeeds.
    """
    hit_rolls = rng.integers(1, 7, size=attacks)
    hits = int(
        np.sum((hit_rolls != 1) & ((hit_rolls >= ballistic_skill) | (hit_rolls == 6)))
    )

    if hits == 0:
        return ShootingResult(hits=0, wounds=0, unsaved=0, damage_dealt=0)

    threshold = wound_roll_threshold(strength, target_toughness)
    wound_rolls = rng.integers(1, 7, size=hits)
    wounds = int(
        np.sum((wound_rolls != 1) & ((wound_rolls >= threshold) | (wound_rolls == 6)))
    )

    if wounds == 0:
        return ShootingResult(hits=hits, wounds=0, unsaved=0, damage_dealt=0)

    modified_save = target_save + ap
    save_rolls = rng.integers(1, 7, size=wounds)
    saves = int(np.sum((save_rolls != 1) & (save_rolls >= modified_save)))
    unsaved = wounds - saves

    if unsaved <= 0:
        return ShootingResult(hits=hits, wounds=wounds, unsaved=0, damage_dealt=0)

    damage_dealt = unsaved * damage
    return ShootingResult(
        hits=hits, wounds=wounds, unsaved=unsaved, damage_dealt=damage_dealt
    )


def expected_damage(
    attacks: int,
    ballistic_skill: int,
    strength: int,
    ap: int,
    damage: int,
    target_toughness: int,
    target_save: int,
) -> float:
    """Closed-form analytical expected damage for one model shooting at one target."""
    p_hit = (7 - ballistic_skill) / 6.0
    p_wound = (7 - wound_roll_threshold(strength, target_toughness)) / 6.0
    modified_save = target_save + ap
    p_save = max(0.0, (7 - modified_save) / 6.0) if modified_save <= 6 else 0.0
    p_fail_save = 1.0 - p_save
    return attacks * p_hit * p_wound * p_fail_save * damage
