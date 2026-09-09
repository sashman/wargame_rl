"""Expected damage for ranged attacks, per pair and as a matrix over stat lines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wargame_rl.wargame.envs.domain.attacks.expectation import expected_attack_damage
from wargame_rl.wargame.envs.domain.attacks.stats import DefenderStats, WeaponStats


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
