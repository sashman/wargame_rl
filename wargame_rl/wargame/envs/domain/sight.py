"""Whether one point can see another: what blocks sight, composed with how it is traced.

This is deliberately separate from `los.py`. That module is the geometry
primitive — a sampled ray against axis-aligned rectangles — and knows nothing
about terrain or the game's blocking rules. This module is the *answer*, built
by handing that primitive the two things that actually block a shot: the static
`blocking_mask` from config, and the terrain footprints that do not contain
either endpoint.

Keeping the composition here rather than on the env facade gives the question
"can A see B?" a single home in the domain.

**The batch entry point is the real one.** `line_of_sight_matrix` traces every
requested pair in one vectorised pass; `has_line_of_sight_between_points` is a
convenience wrapper for the renderer and for tests. Calling the single-pair form
in a loop is the shape that measured a 3x regression, so the shooting mask and
the exposure scan take the matrix.

Terrain is read at call time, never cached. `Battle.set_terrain` replaces the
layout between episodes and a cache here would need invalidating; the footprint
scan is cheap next to the ray.
"""

from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs.domain.los import points_inside_rects, segments_are_clear
from wargame_rl.wargame.envs.domain.terrain import Terrain

BlockingMask = list[list[bool]]


def footprint_bounds(terrain: Terrain) -> np.ndarray:
    """``(M, 4)`` array of footprint rectangles, for the vectorised ray."""
    if not terrain.footprints:
        return np.zeros((0, 4), dtype=float)
    return np.array([(f.x0, f.y0, f.x1, f.y1) for f in terrain.footprints], dtype=float)


def opaque_cell_grid(blocking_mask: BlockingMask | None) -> np.ndarray | None:
    """Convert the config's nested-list blocking mask into a ``[y][x]`` array."""
    if blocking_mask is None:
        return None
    return np.asarray(blocking_mask, dtype=bool)


def line_of_sight_matrix(
    origins: np.ndarray,
    targets: np.ndarray,
    terrain: Terrain,
    blocking_mask: BlockingMask | None,
    *,
    sample_step: float,
    candidates: np.ndarray | None = None,
) -> np.ndarray:
    """``(P, Q)`` — True where origin p can see target q.

    Args:
        origins: ``(P, 2)`` observer positions.
        targets: ``(Q, 2)`` observed positions.
        candidates: optional ``(P, Q)`` bool mask of pairs worth tracing. Pairs
            outside it come back False without being traced at all, which is
            what keeps the cost proportional to the pairs a caller can actually
            use — range gating already rules most of them out.

    Symmetry is exact by construction: the pair (a, b) and the pair (b, a) sample
    the same parametric positions on the same segment, so the same blockers are
    tested. Several metrics depend on that — `firepower_ratio` reads an exposed
    model as one that can also fire.
    """
    n_origins, n_targets = len(origins), len(targets)
    result = np.zeros((n_origins, n_targets), dtype=bool)
    if n_origins == 0 or n_targets == 0:
        return result

    if candidates is None:
        grid_rows, grid_cols = np.meshgrid(
            np.arange(n_origins), np.arange(n_targets), indexing="ij"
        )
        rows, cols = grid_rows.ravel(), grid_cols.ravel()
    else:
        rows, cols = np.nonzero(candidates)
    if len(rows) == 0:
        return result

    starts = np.asarray(origins, dtype=float)[rows]
    ends = np.asarray(targets, dtype=float)[cols]

    blockers = footprint_bounds(terrain)
    exempt: np.ndarray | None = None
    if len(blockers):
        # The see-out / see-into rule (`docs/rules/13-terrain.md`): a piece
        # containing either endpoint does not block, because a model can see out
        # of the ruin it stands in and can be seen while standing in one.
        exempt = points_inside_rects(starts, blockers) | points_inside_rects(
            ends, blockers
        )

    clear = segments_are_clear(
        starts,
        ends,
        blockers,
        sample_step=sample_step,
        blocker_exempt=exempt,
        opaque_cells=opaque_cell_grid(blocking_mask),
    )
    result[rows, cols] = clear
    return result


def has_line_of_sight_between_points(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    terrain: Terrain,
    blocking_mask: BlockingMask | None,
    *,
    sample_step: float,
) -> bool:
    """True when sight is clear between two points. Single-pair convenience."""
    matrix = line_of_sight_matrix(
        np.array([[x0, y0]], dtype=float),
        np.array([[x1, y1]], dtype=float),
        terrain,
        blocking_mask,
        sample_step=sample_step,
    )
    return bool(matrix[0, 0])
