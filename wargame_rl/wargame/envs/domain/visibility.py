"""Continuous line of sight by sampled ray, with model bases as occluders.

Positions are continuous, so sight is traced along the segment between two models
rather than along a chain of grid cells. The segment is sampled at a fixed step and
every sample is tested against the blockers at once, vectorised -- a per-point Python
loop over a 24-inch ray would dominate the step time, and line of sight is the
hottest path in the environment.

Because models occupy space, sight between two of them is not one line but a pencil of
them. Three are traced: centre to centre, and the two outer tangents. That gives the
three-state answer the rules need --

* **hidden** -- no line is clear,
* **visible** -- some line is clear,
* **fully visible** -- every line is clear,

-- and cover keys off the middle case: a target that is visible but not *fully* visible
is in cover.

**Thin features leak.** A blocker narrower than the sample step can fall between two
samples and fail to block. ``WargameEnvConfig`` rejects terrain thinner than the step
for exactly this reason.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class Visibility(IntEnum):
    """How much of a target can be seen. Ordered, so comparisons read naturally."""

    HIDDEN = 0
    VISIBLE = 1
    FULLY_VISIBLE = 2

    @property
    def is_visible(self) -> bool:
        """True when at least one line to the target is clear."""
        return self is not Visibility.HIDDEN

    @property
    def has_cover(self) -> bool:
        """True when the target is seen but not fully -- the cover condition."""
        return self is Visibility.VISIBLE


DEFAULT_SAMPLE_STEP = 0.25


def _sample_points(
    starts: np.ndarray, ends: np.ndarray, step: float
) -> tuple[np.ndarray, int]:
    """Sample every ray at a fixed spacing, returning ``(n_rays, n_samples, 2)``.

    Endpoints are excluded: a model standing inside a terrain feature can still see
    out of it, and one inside another's base would otherwise block itself.
    """
    lengths = np.linalg.norm(ends - starts, axis=1)
    n_samples = max(1, int(np.ceil(float(lengths.max()) / step)))
    # Interior fractions only, so neither endpoint is ever sampled.
    fractions = (np.arange(n_samples, dtype=float) + 0.5) / n_samples
    points = starts[:, None, :] + fractions[None, :, None] * (ends - starts)[:, None, :]
    return points, n_samples


def _inside_rectangles(points: np.ndarray, rectangles: np.ndarray) -> np.ndarray:
    """``(n_rays, n_samples, n_rects)`` bool: which samples fall in which rectangle."""
    px = points[:, :, 0][:, :, None]
    py = points[:, :, 1][:, :, None]
    inside: np.ndarray = (
        (px >= rectangles[:, 0])
        & (px <= rectangles[:, 2])
        & (py >= rectangles[:, 1])
        & (py <= rectangles[:, 3])
    )
    return inside


def _blocked_by_rectangles(points: np.ndarray, rectangles: np.ndarray) -> np.ndarray:
    """``(n_rays,)`` bool: does any sample of each ray fall inside any rectangle."""
    if rectangles.size == 0:
        return np.zeros(points.shape[0], dtype=bool)
    blocked: np.ndarray = np.atleast_1d(
        _inside_rectangles(points, rectangles).any(axis=(1, 2))
    )
    return blocked


def _blocked_by_polygons(points: np.ndarray, padded_vertices: np.ndarray) -> np.ndarray:
    """``(n_rays,)`` bool: does any sample of each ray fall inside any polygon.

    ``padded_vertices`` is ``(n_polygons, n_vertices, 2)``, every outline padded to a
    common vertex count by repeating its last vertex. Padding adds zero-length edges,
    which never straddle a sample's y and so contribute no crossings -- that is what
    lets outlines of different vertex counts share one array.

    Every ray, sample, polygon and edge is evaluated in a single pass. A per-polygon
    Python loop here cost three times the whole step: line of sight runs hundreds of
    queries per phase, and the loop turned each one into dozens of tiny numpy calls
    whose overhead dwarfed the arithmetic.
    """
    n_rays = points.shape[0]
    if padded_vertices.size == 0:
        return np.zeros(n_rays, dtype=bool)

    # A polygon can only block if it holds a sample, so anything whose bounding box
    # misses the samples' bounding box is dropped first. A weapon-range ray crosses a
    # small part of the board, so this usually discards most of the layout and shrinks
    # the array the crossing test below has to build.
    low = points.reshape(-1, 2).min(axis=0)
    high = points.reshape(-1, 2).max(axis=0)
    poly_low = padded_vertices.min(axis=1)
    poly_high = padded_vertices.max(axis=1)
    near = ~(
        (poly_high[:, 0] < low[0])
        | (poly_low[:, 0] > high[0])
        | (poly_high[:, 1] < low[1])
        | (poly_low[:, 1] > high[1])
    )
    padded_vertices = padded_vertices[near]
    if padded_vertices.size == 0:
        return np.zeros(n_rays, dtype=bool)

    px = points[:, :, None, None, 0]  # (rays, samples, 1, 1)
    py = points[:, :, None, None, 1]
    ax = padded_vertices[None, None, :, :, 0]  # (1, 1, polys, verts)
    ay = padded_vertices[None, None, :, :, 1]
    bx = np.roll(ax, -1, axis=3)
    by = np.roll(ay, -1, axis=3)

    straddles = (ay > py) != (by > py)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_at_py = (bx - ax) * (py - ay) / (by - ay) + ax
    crossings = straddles & (px < x_at_py)
    inside = (crossings.sum(axis=3) % 2) == 1  # (rays, samples, polys)
    blocked: np.ndarray = np.atleast_1d(inside.any(axis=(1, 2)))
    return blocked


def _blocked_by_discs(
    points: np.ndarray, centres: np.ndarray, radii: np.ndarray
) -> np.ndarray:
    """``(n_rays,)`` bool: does any sample of each ray fall inside any disc."""
    if centres.size == 0:
        return np.zeros(points.shape[0], dtype=bool)
    deltas = points[:, :, None, :] - centres[None, None, :, :]
    squared = np.einsum("rsod,rsod->rso", deltas, deltas)
    blocked: np.ndarray = np.atleast_1d(
        (squared < radii[None, None, :] ** 2).any(axis=(1, 2))
    )
    return blocked


def _excluding_endpoints(
    centres: np.ndarray,
    radii: np.ndarray,
    observer: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop occluders whose base covers either endpoint of the query."""
    if centres.size == 0:
        return centres, radii
    covers_observer = (
        np.einsum("od,od->o", centres - observer, centres - observer) <= radii**2
    )
    covers_target = (
        np.einsum("od,od->o", centres - target, centres - target) <= radii**2
    )
    keep = ~(covers_observer | covers_target)
    return centres[keep], radii[keep]


def visibility_between(
    observer: np.ndarray,
    target: np.ndarray,
    observer_radius: float,
    target_radius: float,
    polygon_vertices: np.ndarray,
    rectangles: np.ndarray,
    occluder_centres: np.ndarray,
    occluder_radii: np.ndarray,
    step: float = DEFAULT_SAMPLE_STEP,
) -> Visibility:
    """Classify how much of *target* is visible from *observer*.

    ``polygon_vertices`` are the padded terrain outlines that can block this query --
    the caller filters out any the endpoints stand in, so a model inside a ruin still
    sees out of it. ``rectangles`` is the
    legacy board-sized blocking mask, which is opaque matter rather than shelter and so
    is never filtered. ``occluder_centres`` and ``occluder_radii`` are the bases of
    other models that block sight; per the rules the caller excludes models in the
    observer's own unit and in the target's.
    """
    direction = target - observer
    length = float(np.linalg.norm(direction))
    if length == 0.0:
        return Visibility.FULLY_VISIBLE

    # Perpendicular offsets pick out the two outer tangents of the pencil of lines
    # between the two bases.
    perpendicular = np.array([-direction[1], direction[0]]) / length
    starts = np.array(
        [
            observer,
            observer + perpendicular * observer_radius,
            observer - perpendicular * observer_radius,
        ]
    )
    ends = np.array(
        [
            target,
            target + perpendicular * target_radius,
            target - perpendicular * target_radius,
        ]
    )

    # Drop any occluder whose base covers an endpoint -- the same rule that lets a
    # model see out of the ruin it stands in. Without it a model's own base sits on
    # the start of every ray it casts and blocks all of them.
    centres, radii = _excluding_endpoints(
        occluder_centres, occluder_radii, observer, target
    )

    points, _ = _sample_points(starts, ends, step)
    blocked = (
        _blocked_by_rectangles(points, rectangles)
        | _blocked_by_polygons(points, polygon_vertices)
        | _blocked_by_discs(points, centres, radii)
    )

    if not blocked.any():
        return Visibility.FULLY_VISIBLE
    if blocked.all():
        return Visibility.HIDDEN
    return Visibility.VISIBLE
