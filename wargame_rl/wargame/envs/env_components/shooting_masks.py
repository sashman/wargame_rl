"""Per-model shooting target validity masks.

Pure function: takes positions, alive masks, LOS callable, and weapon ranges;
returns (n_player, n_opponent) boolean mask for which targets each model can shoot.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np


def compute_shooting_masks(
    player_positions: np.ndarray,
    opponent_positions: np.ndarray,
    player_alive: np.ndarray,
    opponent_alive: np.ndarray,
    player_max_ranges: np.ndarray,
    has_los_fn: Callable[[int, int, int, int], bool],
    *,
    player_advanced: np.ndarray | None = None,
    engagement_range: float = 0.0,
) -> np.ndarray:
    """Per-model shooting validity: ``(n_player, n_opponent)`` bool mask.

    A target K is valid for model M iff:
    - M is alive (player_alive[M] is True)
    - M has not advanced this turn (player_advanced[M] is not True)
    - M is not within engagement_range of any opponent
    - K is alive (opponent_alive[K] is True)
    - Euclidean distance(M, K) <= player_max_ranges[M]
    - has_los_fn(Mx, My, Kx, Ky) is True

    Models with player_max_ranges <= 0 (no weapons) cannot shoot anyone.
    """
    n_player = len(player_positions)
    n_opponent = len(opponent_positions)
    mask = np.zeros((n_player, n_opponent), dtype=bool)

    if n_opponent == 0:
        return mask

    deltas = player_positions[:, np.newaxis, :] - opponent_positions[np.newaxis, :, :]
    distances = np.linalg.norm(deltas, axis=2)  # (n_player, n_opponent)

    for m in range(n_player):
        if not player_alive[m] or player_max_ranges[m] <= 0:
            continue
        if player_advanced is not None and player_advanced[m]:
            continue
        if engagement_range > 0 and float(distances[m].min()) <= engagement_range:
            continue
        mx, my = int(player_positions[m, 0]), int(player_positions[m, 1])
        for k in range(n_opponent):
            if not opponent_alive[k]:
                continue
            if distances[m, k] > player_max_ranges[m]:
                continue
            kx, ky = int(opponent_positions[k, 0]), int(opponent_positions[k, 1])
            if has_los_fn(mx, my, kx, ky):
                mask[m, k] = True
    return mask


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
    has_los_fn: Callable[[int, int, int, int], bool],
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

    for m in range(n_player):
        if not player_alive[m]:
            continue
        mx, my = int(player_positions[m, 0]), int(player_positions[m, 1])
        for k in range(n_opponent):
            if not opponent_alive[k]:
                continue
            distance = distances[m, k]
            # The `> 0` guards matter: an unarmed model has range 0.0, and
            # `0 <= 0` would make two models on the same cell threaten each
            # other with weapons neither of them has.
            player_reaches = (
                player_max_ranges[m] > 0 and distance <= player_max_ranges[m]
            )
            opponent_reaches = (
                opponent_max_ranges[k] > 0 and distance <= opponent_max_ranges[k]
            )
            if not (player_reaches or opponent_reaches):
                continue
            kx, ky = int(opponent_positions[k, 0]), int(opponent_positions[k, 1])
            if not has_los_fn(mx, my, kx, ky):
                continue
            if opponent_reaches:
                threat_to_player[m] += 1
                opponent_can_shoot[k] = True
            if player_reaches:
                threat_to_opponent[k] += 1
                player_can_shoot[m] = True

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
