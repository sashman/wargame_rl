"""Per-model shooting target validity masks.

Pure function: takes positions, alive masks, LOS callable, and weapon ranges;
returns (n_player, n_opponent) boolean mask for which targets each model can shoot.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def compute_shooting_masks(
    player_positions: np.ndarray,
    opponent_positions: np.ndarray,
    player_alive: np.ndarray,
    opponent_alive: np.ndarray,
    player_max_ranges: np.ndarray,
    has_los_fn: Callable[[int, int, int, int], bool],
) -> np.ndarray:
    """Per-model shooting validity: ``(n_player, n_opponent)`` bool mask.

    A target K is valid for model M iff:
    - M is alive (player_alive[M] is True)
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
