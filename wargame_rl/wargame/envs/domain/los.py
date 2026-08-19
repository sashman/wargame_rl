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

    Absolute offsets alone are not enough to make the answer a property of the
    pair: they are measured from the segment's *start*, so the caller's choice
    of which endpoint comes first still moved the samples. `segments_are_clear`
    orders the endpoints canonically before calling this, which closes that.

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

    # Sight is a property of the PAIR, and samples sit at absolute distances
    # from the segment's start -- so tracing A->B and B->A sampled different
    # points whenever the length was not a whole number of steps, and the two
    # directions could disagree. Ordering the endpoints canonically makes the
    # sample set independent of which one the caller passed first. The rest of
    # the pass is direction-agnostic: `blocker_exempt` is per segment, and the
    # result is a single "clear" flag, so swapping changes nothing else.
    swap = (starts[:, 0] > ends[:, 0]) | (
        (starts[:, 0] == ends[:, 0]) & (starts[:, 1] > ends[:, 1])
    )
    keep = swap[:, np.newaxis]
    starts, ends = np.where(keep, ends, starts), np.where(keep, starts, ends)

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
