"""Line-of-sight exposure and terrain-proximity statistics.

Measurement only — nothing here feeds back into the game. It exists because
"did the agent take cover?" is otherwise unanswerable: terrain blocks line of
sight and nothing else, so cover use shows up as a drop in how often an enemy
can see you, and in how far the army stays from the nearest ruin.
"""

from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs.domain.terrain import Footprint


def distances_to_nearest_footprint(
    positions: np.ndarray,
    footprints: list[Footprint],
) -> np.ndarray:
    """Euclidean distance from each position to the nearest footprint, 0 inside.

    Returns an array of ``inf`` when there is no terrain, so callers that
    average over it do not silently report 0 (which would read as "hugging
    cover") on a board that has none.
    """
    n = len(positions)
    if n == 0:
        return np.zeros(0, dtype=float)
    if not footprints:
        return np.full(n, np.inf, dtype=float)

    rects = np.array(
        [(f.x0, f.y0, f.x1, f.y1) for f in footprints], dtype=float
    )  # (n_rects, 4)
    px = positions[:, 0][:, np.newaxis]
    py = positions[:, 1][:, np.newaxis]

    # Distance from a point to an axis-aligned rectangle: the per-axis overshoot
    # past the nearer edge, zero when the point is inside that axis' span.
    dx = np.maximum(np.maximum(rects[:, 0] - px, px - rects[:, 2]), 0.0)
    dy = np.maximum(np.maximum(rects[:, 1] - py, py - rects[:, 3]), 0.0)
    nearest: np.ndarray = np.hypot(dx, dy).min(axis=1)
    return nearest


class ExposureTracker:
    """Accumulates exposure and terrain proximity over one episode.

    Sampled once per shooting phase rather than per step, so the two numbers
    describe the same moments and stay comparable to each other.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated counts. Called at the start of each episode."""
        self._exposed_models = 0
        self._sampled_models = 0
        self._distance_sum = 0.0
        self._distance_samples = 0

    def record(
        self,
        exposed: np.ndarray,
        alive: np.ndarray,
        terrain_distances: np.ndarray,
    ) -> None:
        """Record one shooting phase.

        Args:
            exposed: per player model, True if any enemy has LOS and range to it.
            alive: per player model alive mask; dead models are not counted.
            terrain_distances: per player model distance to the nearest footprint.
        """
        self._exposed_models += int((exposed & alive).sum())
        self._sampled_models += int(alive.sum())

        finite = np.isfinite(terrain_distances) & alive
        self._distance_sum += float(terrain_distances[finite].sum())
        self._distance_samples += int(finite.sum())

    @property
    def exposure_rate(self) -> float | None:
        """Fraction of alive model-phases where an enemy could see and shoot.

        None when nothing was sampled — tracking off, or no opponents. Returning
        0.0 there would read as "never exposed", which is the opposite of
        "not measured".
        """
        if self._sampled_models == 0:
            return None
        return self._exposed_models / self._sampled_models

    @property
    def terrain_proximity(self) -> float | None:
        """Mean distance from an alive model to the nearest terrain footprint.

        None when nothing was sampled or the board has no terrain — 0.0 would
        read as "standing in cover".
        """
        if self._distance_samples == 0:
            return None
        return self._distance_sum / self._distance_samples
