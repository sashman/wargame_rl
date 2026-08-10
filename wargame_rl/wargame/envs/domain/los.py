"""Line-of-sight geometry: sampled rays against polygon blockers.

The board is continuous, so sight is traced by sampling points along the segment
rather than by walking cells, and blockers are *outlines* rather than bounding
boxes. ``sample_step`` is the spacing guarantee: a blocker thinner than it can
fall between two samples and leak sight, which is why the config validator
rejects terrain narrower than the step.

**Everything here is vectorised over segments *and* over blockers, and that is
not a micro-optimisation.** Sight is the hottest thing in the environment — a
25v25 shooting phase asks about hundreds of pairs — and a per-blocker python loop
turns each query into dozens of tiny numpy calls whose overhead dwarfs the
arithmetic. The prototype measured that shape at 70.2 ms/step against a single
vectorised pass at 30.0 ms.

This module is the geometry primitive and knows nothing about terrain or the
game's blocking rules. `sight.py` composes it with them.
"""

from __future__ import annotations

import numpy as np

from wargame_rl.wargame.envs.types.geometry import polygons_contain_points

# Cap on the (segments x samples x blockers) working set of one pass, in
# elements. Segments are processed in chunks small enough to respect it, so
# memory stays flat as the army or the layout grows.
_MAX_WORKING_ELEMENTS = 4_000_000


def _interior_sample_offsets(
    lengths: np.ndarray, sample_step: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(N, T)`` parametric sample positions per segment, and which are real.

    Samples sit at **absolute** distances ``k * sample_step`` along each segment,
    so a pair's answer depends only on that pair. The obvious cheaper scheme --
    one shared set of parametric offsets sized from the longest segment in the
    batch -- makes the answer depend on *what else was asked at the same time*:
    split one batch into two and the sample positions move, so a ray can start
    or stop being blocked. That was a real defect, and it surfaced as two golden
    gates failing when a caller began tracing in two passes instead of one.

    The array is still rectangular, sized by the longest segment, with a mask
    marking the samples that fall inside each segment. Endpoints are excluded: a
    model standing on a blocker is handled by the see-out rule in `sight.py`,
    not by the ray.
    """
    if sample_step <= 0:
        raise ValueError(f"sample_step must be positive, got {sample_step}")
    max_intervals = max(1, int(np.ceil(float(lengths.max()) / sample_step)))
    steps = np.arange(1, max_intervals, dtype=float) * sample_step  # (T,)
    safe = np.where(lengths > 0, lengths, 1.0)
    offsets = steps[np.newaxis, :] / safe[:, np.newaxis]  # (N, T)
    return offsets, offsets < 1.0


def segments_are_clear(
    starts: np.ndarray,
    ends: np.ndarray,
    outlines: np.ndarray,
    vertex_counts: np.ndarray,
    *,
    sample_step: float,
    blocker_exempt: np.ndarray | None = None,
    opaque_cells: np.ndarray | None = None,
) -> np.ndarray:
    """Trace ``N`` segments at once. Returns ``(N,)`` — True where sight is clear.

    Args:
        starts: ``(N, 2)`` segment origins.
        ends: ``(N, 2)`` segment endpoints.
        outlines: ``(M, V, 2)`` blocker outlines, padded to a common vertex
            budget by repeating the last vertex.
        vertex_counts: ``(M,)`` real vertex count per outline.
        sample_step: maximum spacing between consecutive samples, in board units.
        blocker_exempt: ``(N, M)`` — True where that blocker does not block that
            segment. This is how the see-out rule enters: it is *per query*,
            because whether a piece blocks depends on where the endpoints are.
        opaque_cells: ``(height, width)`` static mask of impassable-to-sight
            cells, indexed ``[y][x]``. Gets no exemption — it is opaque matter
            rather than shelter, so standing beside it does not let you see
            through it.
    """
    n_segments = len(starts)
    clear = np.ones(n_segments, dtype=bool)
    if n_segments == 0:
        return clear

    deltas = ends - starts
    lengths = np.linalg.norm(deltas, axis=1)
    offsets, real_sample = _interior_sample_offsets(lengths, sample_step)
    n_samples = offsets.shape[1]
    if n_samples == 0:
        return clear

    n_blockers = len(outlines)
    vertex_budget = outlines.shape[1] if n_blockers else 1
    # The membership test materialises (points x blockers x edges), so the edge
    # budget belongs in the chunk size. Leaving it out is how a layout of
    # many-sided outlines turns a bounded working set into a multi-gigabyte one.
    per_segment = max(1, n_samples * max(1, n_blockers) * max(1, vertex_budget))
    chunk = max(1, _MAX_WORKING_ELEMENTS // per_segment)

    for lo in range(0, n_segments, chunk):
        hi = min(lo + chunk, n_segments)
        span = hi - lo
        # (span, samples, 2)
        chunk_offsets = offsets[lo:hi]
        chunk_real = real_sample[lo:hi]
        points = (
            starts[lo:hi, np.newaxis, :]
            + deltas[lo:hi, np.newaxis, :] * chunk_offsets[:, :, np.newaxis]
        )
        blocked = np.zeros(span, dtype=bool)

        if n_blockers:
            inside = polygons_contain_points(
                points.reshape(-1, 2), outlines, vertex_counts
            ).reshape(span, n_samples, n_blockers)
            inside &= chunk_real[:, :, np.newaxis]
            if blocker_exempt is not None:
                inside &= ~blocker_exempt[lo:hi, np.newaxis, :]
            blocked |= inside.any(axis=(1, 2))

        if opaque_cells is not None:
            height, width = opaque_cells.shape
            ix = np.clip(points[:, :, 0].astype(np.intp), 0, width - 1)
            iy = np.clip(points[:, :, 1].astype(np.intp), 0, height - 1)
            blocked |= (opaque_cells[iy, ix] & chunk_real).any(axis=1)

        clear[lo:hi] = ~blocked

    return clear


def segments_clear_of_discs(
    starts: np.ndarray,
    ends: np.ndarray,
    centres: np.ndarray,
    radii: np.ndarray,
    *,
    exempt: np.ndarray | None = None,
) -> np.ndarray:
    """``(N,)`` — True where no disc blocks the segment. Exact, not sampled.

    Model bases are circles, and a circle has a closed-form segment test: the
    perpendicular distance from its centre to the segment. So unlike terrain
    this needs no sampling at all, and carries no `sample_step` caveat — a model
    thinner than the sample step could never leak sight the way a thin ruin can.

    Args:
        exempt: ``(N, M)`` — True where that disc does not block that segment.
            This is where the unit rule enters: a model ignores others in its
            own unit and in its target's unit.
    """
    n_segments = len(starts)
    clear = np.ones(n_segments, dtype=bool)
    if n_segments == 0 or len(centres) == 0:
        return clear

    # Written out in components on purpose. The readable form -- build the
    # closest point, then `np.linalg.norm` -- allocates two (N, M, 2) arrays and
    # takes a square root over (N, M), and sight is the hottest thing in the
    # environment. This compares squared distances and never materialises a
    # point.
    # No bounding-box pre-filter here, deliberately. The same trick is worth 8x
    # on the polygon test, where the inner work is (points x shapes x edges)
    # with a division in it. A disc test has no edge dimension and no division,
    # so the filter's own four comparisons cost more than the projection they
    # avoid: measured 8.18 -> 10.21 ms/step with one in front.
    delta_x = ends[:, 0] - starts[:, 0]
    delta_y = ends[:, 1] - starts[:, 1]
    length_sq = delta_x**2 + delta_y**2
    safe_length = np.where(length_sq > 0, length_sq, 1.0)

    offset_x = centres[np.newaxis, :, 0] - starts[:, np.newaxis, 0]  # (N, M)
    offset_y = centres[np.newaxis, :, 1] - starts[:, np.newaxis, 1]
    along = np.clip(
        (offset_x * delta_x[:, np.newaxis] + offset_y * delta_y[:, np.newaxis])
        / safe_length[:, np.newaxis],
        0.0,
        1.0,
    )
    gap_x = offset_x - along * delta_x[:, np.newaxis]
    gap_y = offset_y - along * delta_y[:, np.newaxis]

    blocking = (gap_x**2 + gap_y**2) < (radii**2)[np.newaxis, :]
    if exempt is not None:
        blocking &= ~exempt
    unblocked: np.ndarray = ~blocking.any(axis=1)
    return unblocked
