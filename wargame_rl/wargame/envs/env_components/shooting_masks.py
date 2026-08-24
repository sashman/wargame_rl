"""Per-model shooting target validity masks.

Pure function: takes positions, alive masks, LOS callable, and weapon ranges;
returns (n_player, n_opponent) boolean mask for which targets each model can shoot.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from wargame_rl.wargame.envs.domain.engagement import engaged_units, engaged_with_any

# Trace sight for many pairs at once: given ``(P, 2)`` origins, ``(Q, 2)``
# targets and a ``(P, Q)`` mask of pairs worth tracing, return ``(P, Q)`` of
# which are clear. A per-pair callable would be the natural signature and is the
# wrong one -- sight is the hot path, and asking it one pair at a time turns each
# query into a handful of tiny numpy calls whose overhead dwarfs the arithmetic.
LosMatrixFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def compute_shooting_masks(
    player_positions: np.ndarray,
    opponent_positions: np.ndarray,
    player_alive: np.ndarray,
    opponent_alive: np.ndarray,
    player_max_ranges: np.ndarray,
    los_matrix_fn: LosMatrixFn,
    *,
    player_advanced: np.ndarray | None = None,
    engagement_range: float = 0.0,
    base_diameter: float = 0.0,
) -> np.ndarray:
    """Per-model shooting validity: ``(n_player, n_opponent)`` bool mask.

    A target K is valid for model M iff:
    - M is alive (player_alive[M] is True)
    - M has not advanced this turn (player_advanced[M] is not True)
    - M is not within engagement_range of any opponent
    - K is alive (opponent_alive[K] is True)
    - Euclidean distance(M, K) <= player_max_ranges[M]
    - sight is clear from M to K

    Models with player_max_ranges <= 0 (no weapons) cannot shoot anyone.

    Every cheap condition is applied *before* sight, and the survivors are traced
    in one batch. Sight is by far the most expensive term, so narrowing the pair
    set first is what keeps it affordable.
    """
    n_player = len(player_positions)
    n_opponent = len(opponent_positions)
    mask = np.zeros((n_player, n_opponent), dtype=bool)

    if n_opponent == 0 or n_player == 0:
        return mask

    deltas = player_positions[:, np.newaxis, :] - opponent_positions[np.newaxis, :, :]
    distances = np.linalg.norm(deltas, axis=2)  # (n_player, n_opponent)

    shooters = np.asarray(player_alive, dtype=bool) & (player_max_ranges > 0)
    if player_advanced is not None:
        shooters &= ~np.asarray(player_advanced, dtype=bool)
    alive_targets = np.asarray(opponent_alive, dtype=bool)
    if engagement_range > 0:
        shooters &= ~engaged_with_any(
            player_positions,
            opponent_positions,
            alive_targets,
            np.asarray(player_alive, dtype=bool),
            engagement_range=engagement_range,
            base_diameter=base_diameter,
        )

    candidates = (
        shooters[:, np.newaxis]
        & alive_targets[np.newaxis, :]
        & (distances <= player_max_ranges[:, np.newaxis])
    )
    if not candidates.any():
        return mask

    visible: np.ndarray = candidates & los_matrix_fn(
        player_positions, opponent_positions, candidates
    )
    return visible


def shooter_range_mask(
    player_positions: np.ndarray,
    opponent_positions: np.ndarray,
    opponent_alive: np.ndarray,
    player_max_ranges: np.ndarray,
) -> np.ndarray:
    """``(n_player, n_opponent)`` where a live target is within weapon range.

    Range alone, with no sight term. The rules check range and visibility
    independently against the target *unit*, so the two have to stay separable
    up to the point where they are combined per unit.
    """
    n_player, n_opponent = len(player_positions), len(opponent_positions)
    if n_player == 0 or n_opponent == 0:
        return np.zeros((n_player, n_opponent), dtype=bool)
    deltas = player_positions[:, np.newaxis, :] - opponent_positions[np.newaxis, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    within: np.ndarray = (distances <= player_max_ranges[:, np.newaxis]) & np.asarray(
        opponent_alive, dtype=bool
    )[np.newaxis, :]
    return within


class ThreatCounts(NamedTuple):
    """Who threatens whom, and who has a target, from one line-of-sight scan."""

    threat_to_player: np.ndarray
    """``[M]`` = alive opponents that can see and reach player model M."""

    threat_to_opponent: np.ndarray
    """``[K]`` = alive player models that can see and reach opponent K."""

    player_can_shoot: np.ndarray
    """``[M]`` = True when player model M has at least one reachable target."""

    opponent_can_shoot: np.ndarray
    """``[K]`` = True when opponent K has at least one reachable target."""


def compute_threat_counts(
    player_positions: np.ndarray,
    opponent_positions: np.ndarray,
    player_alive: np.ndarray,
    opponent_alive: np.ndarray,
    player_max_ranges: np.ndarray,
    opponent_max_ranges: np.ndarray,
    los_matrix_fn: LosMatrixFn,
) -> ThreatCounts:
    """Mutual threat counts and per-side "has a target" masks.

    The threat counts and the shoot masks index the *same* pairs differently.
    They are not redundant: ``threat_to_player[M]`` says how many guns bear on
    M, while ``player_can_shoot[M]`` says whether M fires. At equal weapon
    ranges those coincide per model -- line of sight is symmetric, so being
    shootable and being able to shoot are the same condition -- but the counts
    of each still differ across the two sides, and at unequal ranges even the
    per-model identity breaks.

    Deliberately ungated, unlike :func:`compute_shooting_masks`: no ``advanced``
    flag and no engagement-range check. Those two describe whose turn it is
    rather than who is dangerous, and a model in base contact is still a threat
    next round -- gating on them would score a headlong charge as safety.

    All four come out of one line-of-sight scan. They are separate arrays rather
    than one transposed matrix because a transpose is only correct while both
    sides carry the same weapon range.
    """
    n_player = len(player_positions)
    n_opponent = len(opponent_positions)
    threat_to_player = np.zeros(n_player, dtype=np.int64)
    threat_to_opponent = np.zeros(n_opponent, dtype=np.int64)
    player_can_shoot = np.zeros(n_player, dtype=bool)
    opponent_can_shoot = np.zeros(n_opponent, dtype=bool)

    if n_player == 0 or n_opponent == 0:
        return ThreatCounts(
            threat_to_player,
            threat_to_opponent,
            player_can_shoot,
            opponent_can_shoot,
        )

    deltas = player_positions[:, np.newaxis, :] - opponent_positions[np.newaxis, :, :]
    distances = np.linalg.norm(deltas, axis=2)  # (n_player, n_opponent)

    both_alive = (
        np.asarray(player_alive, dtype=bool)[:, np.newaxis]
        & np.asarray(opponent_alive, dtype=bool)[np.newaxis, :]
    )
    # The `> 0` guards matter: an unarmed model has range 0.0, and `0 <= 0`
    # would make two models on the same spot threaten each other with weapons
    # neither of them has.
    player_reaches = both_alive & (
        (player_max_ranges[:, np.newaxis] > 0)
        & (distances <= player_max_ranges[:, np.newaxis])
    )
    opponent_reaches = both_alive & (
        (opponent_max_ranges[np.newaxis, :] > 0)
        & (distances <= opponent_max_ranges[np.newaxis, :])
    )

    candidates = player_reaches | opponent_reaches
    if not candidates.any():
        return ThreatCounts(
            threat_to_player,
            threat_to_opponent,
            player_can_shoot,
            opponent_can_shoot,
        )

    visible = candidates & los_matrix_fn(
        player_positions, opponent_positions, candidates
    )

    threat_to_player = (visible & opponent_reaches).sum(axis=1)
    threat_to_opponent = (visible & player_reaches).sum(axis=0)
    opponent_can_shoot = (visible & opponent_reaches).any(axis=0)
    player_can_shoot = (visible & player_reaches).any(axis=1)

    return ThreatCounts(
        threat_to_player,
        threat_to_opponent,
        player_can_shoot,
        opponent_can_shoot,
    )


def max_weapon_ranges(
    model_configs: list | None,
    n_models: int,
) -> np.ndarray:
    """Max weapon range per model from config. 0.0 for models with no weapons.

    Uses the longest-ranged weapon per model since a target is "in range"
    if any weapon can reach it.
    """
    ranges = np.zeros(n_models, dtype=float)
    if model_configs is None:
        return ranges
    for i, mc in enumerate(model_configs):
        if mc.weapons:
            ranges[i] = max(w.range for w in mc.weapons)
    return ranges


def group_shooting_masks(
    model_target_mask: np.ndarray,
    in_range: np.ndarray,
    target_groups: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    """Reduce a per-model mask to ``(n_shooters, n_groups)`` unit targetability.

    A weapon names an enemy **unit**, and the rules check its two conditions
    *independently*:

        "Visibility and range are checked independently and need not be
        satisfied by the same enemy model -- it is enough that some model in the
        target unit is visible and some model in it is in range."

    So a unit is a legal target when *any* of its models can be seen **and**
    *any* of its models is reachable, even when those are different models. That
    is strictly more permissive than requiring one model to satisfy both, which
    is what a per-model mask does and what OR-ing a per-model mask over the unit
    would preserve.

    Args:
        model_target_mask: ``(n_shooters, n_targets)`` where sight is clear
            *and* the shooter is eligible to fire at all.
        in_range: ``(n_shooters, n_targets)`` where the target is within range,
            regardless of sight.
        target_groups: ``(n_targets,)`` group id per target model.
        n_groups: how many units the target army splits into.
    """
    n_shooters = model_target_mask.shape[0]
    mask = np.zeros((n_shooters, n_groups), dtype=bool)
    if n_shooters == 0 or n_groups == 0 or model_target_mask.shape[1] == 0:
        return mask
    for group in range(n_groups):
        members = target_groups == group
        if not members.any():
            continue
        mask[:, group] = model_target_mask[:, members].any(axis=1) & in_range[
            :, members
        ].any(axis=1)
    return mask


def compute_unit_shooting_masks(
    player_positions: np.ndarray,
    opponent_positions: np.ndarray,
    player_alive: np.ndarray,
    opponent_alive: np.ndarray,
    player_max_ranges: np.ndarray,
    los_matrix_fn: LosMatrixFn,
    target_groups: np.ndarray,
    n_groups: int,
    *,
    player_advanced: np.ndarray | None = None,
    engagement_range: float = 0.0,
    base_diameter: float = 0.0,
    exclude_engaged_targets: bool = False,
) -> np.ndarray:
    """Which enemy **units** each model may declare against: ``(n_player, n_groups)``.

    The whole shooting-target check in one place, because the rules put its two
    halves on different models: a unit is targetable when *some* model in it is
    visible and *some* model in it is in range, and those need not be the same
    model. Reducing a per-model "visible AND in range" mask over the unit would
    quietly keep them coupled and reject legal targets.

    Sight is still gated, just at unit granularity: a pair is traced only when
    the target's unit has *some* model in range of the shooter. That preserves
    the batching the per-model version exists for -- tracing every pair on the
    board is the shape that measured a 3x regression -- while no longer letting
    an out-of-range model hide its unit from a shot the rules allow.

    ⚠ `exclude_engaged_targets` closes a rules gap that has been open since
    shooting shipped. `docs/rules/04-making-attacks.md` requires a target be
    "visible, in range and **unengaged**", and `implementation-status.md` rated
    that row *implemented* -- but the engagement term gated only the SHOOTER.
    Nothing ever reduced the target axis by an engagement test, so a unit locked
    in melee could be shot freely by everyone not themselves engaged.

    It has been invisible rather than wrong: `back_off_to_unengaged` runs on
    every mover on both seats, so engagement is 0.0000% of model-pairs and the
    clause has never had an opportunity to bite. It is therefore off by default
    and a **bit-identical no-op today**, and it is wired now rather than with
    the charge so that the rules correction and the mechanic are separately
    attributable.
    """
    n_player = len(player_positions)
    mask = np.zeros((n_player, n_groups), dtype=bool)
    if n_player == 0 or n_groups == 0 or len(opponent_positions) == 0:
        return mask

    deltas = player_positions[:, np.newaxis, :] - opponent_positions[np.newaxis, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    alive_targets = np.asarray(opponent_alive, dtype=bool)

    shooters = np.asarray(player_alive, dtype=bool) & (player_max_ranges > 0)
    if player_advanced is not None:
        shooters &= ~np.asarray(player_advanced, dtype=bool)
    if engagement_range > 0:
        shooters &= ~engaged_with_any(
            player_positions,
            opponent_positions,
            alive_targets,
            np.asarray(player_alive, dtype=bool),
            engagement_range=engagement_range,
            base_diameter=base_diameter,
        )
    if not shooters.any():
        return mask

    in_range = (distances <= player_max_ranges[:, np.newaxis]) & alive_targets

    members = [target_groups == group for group in range(n_groups)]
    group_in_range = np.zeros((n_player, n_groups), dtype=bool)
    for group, member in enumerate(members):
        if member.any():
            group_in_range[:, group] = in_range[:, member].any(axis=1)

    # Trace only pairs whose *unit* is reachable -- but every model of such a
    # unit, since the visible model need not be the reachable one.
    reachable_member = np.zeros_like(in_range)
    for group, member in enumerate(members):
        if member.any():
            reachable_member[:, member] = group_in_range[:, group][:, np.newaxis]
    candidates = (
        shooters[:, np.newaxis] & alive_targets[np.newaxis, :] & reachable_member
    )
    if not candidates.any():
        return mask

    visible = candidates & los_matrix_fn(
        player_positions, opponent_positions, candidates
    )
    targetable_group = np.ones(n_groups, dtype=bool)
    if exclude_engaged_targets and engagement_range > 0:
        # A unit is engaged when any of its models is -- the rule is unit-level,
        # so one model in contact shields the whole unit from shooting.
        target_engaged = engaged_with_any(
            opponent_positions,
            player_positions,
            np.asarray(player_alive, dtype=bool),
            # ⚠ The subject axis is the OPPONENT here, not the player. Omitting
            # it let an enemy CORPSE beside one of my models shield its whole
            # unit -- the 2026-08-19 corpse defect, on the target axis.
            alive_targets,
            engagement_range=engagement_range,
            base_diameter=base_diameter,
        )
        targetable_group = ~engaged_units(target_engaged, target_groups, n_groups)

    for group, member in enumerate(members):
        if member.any():
            mask[:, group] = (
                visible[:, member].any(axis=1)
                & group_in_range[:, group]
                & targetable_group[group]
            )
    return mask
