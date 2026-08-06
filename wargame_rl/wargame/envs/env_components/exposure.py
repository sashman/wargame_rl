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
        self._our_shooters = 0
        self._their_shooters = 0

    def record(
        self,
        exposed: np.ndarray,
        alive: np.ndarray,
        terrain_distances: np.ndarray,
        our_shooters: int | None = None,
        their_shooters: int | None = None,
    ) -> None:
        """Record one shooting phase.

        Args:
            exposed: per player model, True if any enemy has LOS and range to it.
            alive: per player model alive mask; dead models are not counted.
            terrain_distances: per player model distance to the nearest footprint.
            our_shooters: alive player models with at least one reachable target.
            their_shooters: alive opponents with at least one reachable target.
                Both are needed for the firepower measure; None skips it.
        """
        self._exposed_models += int((exposed & alive).sum())
        self._sampled_models += int(alive.sum())

        finite = np.isfinite(terrain_distances) & alive
        self._distance_sum += float(terrain_distances[finite].sum())
        self._distance_samples += int(finite.sum())

        if our_shooters is not None and their_shooters is not None:
            # Totals, not per-phase ratios: a phase with 20 models firing says
            # more about the exchange than one with 2, and averaging ratios
            # would weight them equally.
            self._our_shooters += int(our_shooters)
            self._their_shooters += int(their_shooters)

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

    @property
    def firepower_ratio(self) -> float | None:
        """(our models that can fire) / (theirs that can), over the episode.

        1.0 is an even exchange; above 1.0 the army brings more guns to bear
        than the enemy does, which is the exchange ratio cover is actually for.
        `exposure_rate` cannot say this: it counts only our side of the trade,
        so hiding and declining a bad fight look identical to hiding and taking
        a good one.

        This counts **shooters**, not targets. The metric began as
        `firepower_advantage`, a difference between the models each side could
        *see*, and that got the direction wrong as well as the arithmetic: line
        of sight is symmetric, so a model that is exposed is exactly a model
        that can fire, and "enemies we can see" is their shooter count, not
        ours. One of our models walking into view of twenty scored twenty --
        which is how `random`, a policy that wins no games, topped the table.
        See the 2026-08-06 report.

        None when nothing was sampled, and when the enemy could never fire at
        all — the ratio is unbounded there, a degenerate case rather than a
        perfect one.
        """
        if self._their_shooters == 0:
            return None
        return self._our_shooters / self._their_shooters
