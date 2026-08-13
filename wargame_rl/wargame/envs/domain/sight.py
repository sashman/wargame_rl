"""Whether one point can see another: what blocks sight, composed with how it is traced.

This is deliberately separate from `los.py`. That module is the geometry
primitive — a sampled ray against axis-aligned rectangles — and knows nothing
about terrain or the game's blocking rules. This module is the *answer*, built
by handing that primitive the two things that actually block a shot: the static
`blocking_mask` from config, and the terrain footprints that do not contain
either endpoint.

Keeping the composition here rather than on the env facade gives the question
"can A see B?" a single home in the domain.

**Models do not block sight — only terrain does.** This is a deliberate
divergence from `docs/rules/06-visibility-and-damage.md`, which lets enemy and
third-party models block: no model on this table has a silhouette that is
actually opaque, since a 32mm infantry figure is a thin shape on a round base
rather than a cylinder, so a line that clips a base can be drawn in practice.
Removing it takes the rules' *ignore your own unit* exemption with it — that
exemption exists only to stop a squad shielding itself from its own occlusion,
so with nothing occluding there is nothing to exempt. `base_radius` still shapes
sight: it sets the width of the corridor traced in `visibility_matrix`, which is
what makes terrain able to grant **cover**.

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

from wargame_rl.wargame.envs.domain.los import segments_are_clear
from wargame_rl.wargame.envs.domain.terrain import Terrain
from wargame_rl.wargame.envs.types.geometry import polygons_contain_points

BlockingMask = list[list[bool]]


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

    outlines = terrain.outlines
    vertex_counts = terrain.vertex_counts
    exempt: np.ndarray | None = None
    if len(outlines):
        # The see-out / see-into rule (`docs/rules/13-terrain.md`): a piece
        # containing either endpoint does not block, because a model can see out
        # of the ruin it stands in and can be seen while standing in one.
        # Edge-inclusive: this is membership deciding a *rule*, so a model
        # standing exactly on a ruin's edge is standing in it and can see out.
        # The samples along the ray use the cheap interior-only test.
        exempt = polygons_contain_points(
            starts, outlines, vertex_counts, include_boundary=True
        ) | polygons_contain_points(
            ends, outlines, vertex_counts, include_boundary=True
        )

    clear = segments_are_clear(
        starts,
        ends,
        outlines,
        vertex_counts,
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


# Three-state visibility, ordered so that `>= COVER` reads as "can be shot at".
HIDDEN = 0
COVER = 1
CLEAR = 2


def visibility_matrix(
    origins: np.ndarray,
    targets: np.ndarray,
    terrain: Terrain,
    blocking_mask: BlockingMask | None,
    *,
    sample_step: float,
    candidates: np.ndarray | None = None,
    origin_radii: np.ndarray | None = None,
    target_radii: np.ndarray | None = None,
) -> np.ndarray:
    """``(P, Q)`` of HIDDEN / COVER / CLEAR — how well origin p sees target q.

    Three rays per pair: the centre line, and two parallel to it offset by the
    wider of the two bases. All three clear is *fully visible*; none clear is
    hidden; anything between is **cover**, which worsens the attack by
    `COVER_RANGED_SKILL_PENALTY`. Only terrain is traced against — see the
    module docstring for why models do not occlude.

    **The offsets are parallel, and symmetric in the pair, on purpose.** The
    literal reading — rays to the two outer tangents of the target — is more
    faithful to "how much of the target can I see", and it is *directional*: the
    rays A casts at B are not the rays B casts at A, so A could see B while B
    could not see A. Several metrics here are built on sight being exactly
    symmetric (`firepower_ratio` reads an exposed model as one that can also
    fire), and an asymmetry would make them silently count two different
    populations. A corridor of the pair's width is symmetric by construction and
    answers the same question closely enough.

    All three rays are traced in one batch rather than three calls — the cost of
    sight is dominated by per-call overhead, not by the arithmetic.

    **This reduces exactly to the two-state answer when models have no base.**
    With `target_radii` all zero the two edge rays coincide with the centre ray,
    so a pair is either CLEAR or HIDDEN and cover can never occur. That is why
    none of this needs a config flag: `base_radius` gates it, and every result
    measured before models had bases is reproduced.
    """
    n_origins, n_targets = len(origins), len(targets)
    result = np.full((n_origins, n_targets), HIDDEN, dtype=np.int8)
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

    # With no radii the two edge rays coincide with the centre one, so tracing
    # them would be three copies of the same query.
    if target_radii is None:
        n_rays, all_starts, all_ends = 1, starts, ends
    else:
        offsets = _corridor_offsets(
            starts, ends, origin_radii, target_radii, rows, cols
        )
        n_rays = 3
        all_starts = np.concatenate([starts, starts + offsets, starts - offsets])
        all_ends = np.concatenate([ends, ends + offsets, ends - offsets])

    clear = _terrain_clear(all_starts, all_ends, terrain, blocking_mask, sample_step)
    n_clear = clear.reshape(n_rays, len(rows)).sum(axis=0)
    result[rows, cols] = np.where(
        n_clear == n_rays, CLEAR, np.where(n_clear == 0, HIDDEN, COVER)
    )
    return result


def _corridor_offsets(
    starts: np.ndarray,
    ends: np.ndarray,
    origin_radii: np.ndarray | None,
    target_radii: np.ndarray | None,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    """``(N, 2)`` perpendicular offsets defining the corridor between each pair.

    Half-width is the *wider* of the two bases, which is symmetric in the pair —
    see `visibility_matrix` for why that matters more here than the literal
    tangent construction.
    """
    if target_radii is None:
        return np.zeros_like(starts)
    widths = np.asarray(target_radii, dtype=float)[cols]
    if origin_radii is not None:
        widths = np.maximum(widths, np.asarray(origin_radii, dtype=float)[rows])
    deltas = ends - starts
    lengths = np.linalg.norm(deltas, axis=1)
    safe = np.where(lengths > 0, lengths, 1.0)
    direction = deltas / safe[:, np.newaxis]
    perpendicular = np.column_stack([-direction[:, 1], direction[:, 0]])
    offsets: np.ndarray = perpendicular * widths[:, np.newaxis]
    return offsets


def _terrain_clear(
    starts: np.ndarray,
    ends: np.ndarray,
    terrain: Terrain,
    blocking_mask: BlockingMask | None,
    sample_step: float,
) -> np.ndarray:
    """Clear-of-terrain for a batch of segments, with the see-out exemption."""
    outlines = terrain.outlines
    vertex_counts = terrain.vertex_counts
    exempt: np.ndarray | None = None
    if len(outlines):
        exempt = polygons_contain_points(
            starts, outlines, vertex_counts, include_boundary=True
        ) | polygons_contain_points(
            ends, outlines, vertex_counts, include_boundary=True
        )
    return segments_are_clear(
        starts,
        ends,
        outlines,
        vertex_counts,
        sample_step=sample_step,
        blocker_exempt=exempt,
        opaque_cells=opaque_cell_grid(blocking_mask),
    )
