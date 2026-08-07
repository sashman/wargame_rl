"""Terrain placement domain service: generate random ruin layouts per episode.

Mirrors `placement.py` in shape — a pure function taking the config, the board
and an rng, returning a domain object. Placement is rejection sampling; the
config validator rejects layouts that pack too tightly for it to converge.

Pieces are convex polygons. A piece is drawn by sampling vertices around an ellipse,
so ``min_size``/``max_size`` still describe the extent of a piece and a four-vertex
draw is a quadrilateral rather than an axis-aligned box. Walls *inside* a ruin are a
separate feature that does not exist yet; this generates outlines only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from wargame_rl.wargame.envs.domain.terrain import Footprint, Terrain
from wargame_rl.wargame.envs.domain.value_objects import BoardDimensions
from wargame_rl.wargame.envs.types.config import RandomTerrainConfig
from wargame_rl.wargame.envs.types.geometry import Polygon

if TYPE_CHECKING:
    from numpy.random import Generator

_MAX_LAYOUT_ATTEMPTS = 50
_MAX_PIECE_ATTEMPTS = 200

# Enough vertices for a piece to read as a shape rather than a box, while staying
# inside the observation's vertex budget.
_MIN_VERTICES = 4
_MAX_VERTICES = 7

# How far a vertex may be pulled in towards the centre, as a fraction of the piece's
# radius. Keeps outlines irregular without producing slivers the sampled ray could
# leak through.
_MIN_RADIUS_FRACTION = 0.7


def _clashes(candidate: Polygon, placed: list[Polygon], min_gap: float) -> bool:
    """True if *candidate* overlaps, or comes within ``min_gap`` of, anything placed.

    The gap is applied by growing the candidate's bounding box, which is a
    conservative test: two pieces whose boxes are within the gap but whose outlines
    are further apart are rejected. That costs a little packing density and buys a
    cheap, predictable check.
    """
    x0, y0, x1, y1 = candidate.bounds
    grown = Polygon.from_rect(x0 - min_gap, y0 - min_gap, x1 + min_gap, y1 + min_gap)
    return any(grown.intersects(other) for other in placed)


def _sample_piece(
    spec: RandomTerrainConfig,
    board: BoardDimensions,
    rng: Generator,
    centre_x: float | None = None,
) -> Polygon:
    """Draw one convex outline uniformly from the region inside the edge margin.

    Vertices are sampled at sorted angles around an ellipse, which guarantees a
    convex, non-self-intersecting outline without needing a hull.
    """
    width = float(rng.uniform(spec.min_size, spec.max_size))
    height = float(rng.uniform(spec.min_size, spec.max_size))

    if centre_x is None:
        low = spec.edge_margin + width / 2.0
        high = board.width - spec.edge_margin - width / 2.0
        centre_x = float(rng.uniform(low, high))
    centre_y = float(
        rng.uniform(
            spec.edge_margin + height / 2.0,
            board.height - spec.edge_margin - height / 2.0,
        )
    )

    n_vertices = int(rng.integers(_MIN_VERTICES, _MAX_VERTICES + 1))
    # Even angular spacing jittered within its own slice keeps the vertices in order
    # around the outline, so the polygon cannot self-intersect.
    slice_width = 2.0 * np.pi / n_vertices
    angles = np.arange(n_vertices) * slice_width + rng.uniform(
        0.0, slice_width, size=n_vertices
    )
    radii = rng.uniform(_MIN_RADIUS_FRACTION, 1.0, size=n_vertices)

    vertices = np.column_stack(
        [
            centre_x + radii * np.cos(angles) * width / 2.0,
            centre_y + radii * np.sin(angles) * height / 2.0,
        ]
    )
    return Polygon(vertices)


def _attempt_layout(
    spec: RandomTerrainConfig,
    board: BoardDimensions,
    rng: Generator,
) -> list[Polygon] | None:
    """Build one candidate layout, or None if a piece could not be placed."""
    placed: list[Polygon] = []

    if spec.mirror and spec.count % 2 == 1:
        # An odd count leaves one piece without a partner, so it has to be its own
        # mirror image. Unlike the old lattice version there is no parity to fix up,
        # because a continuous centre line can be straddled exactly.
        for _ in range(_MAX_PIECE_ATTEMPTS):
            candidate = _sample_centre_piece(spec, board, rng)
            if not _clashes(candidate, placed, spec.min_gap):
                placed.append(candidate)
                break
        else:
            return None

    remaining = spec.count - len(placed)
    n_draws = remaining // 2 if spec.mirror else remaining

    for _ in range(n_draws):
        for _ in range(_MAX_PIECE_ATTEMPTS):
            candidate = _sample_piece(spec, board, rng)
            if not spec.mirror:
                if _clashes(candidate, placed, spec.min_gap):
                    continue
                placed.append(candidate)
                break
            # A piece is checked against its own reflection first — that is what
            # stops a pair from overlapping across the centre line.
            reflection = candidate.mirrored(board.width)
            if _clashes(candidate, [reflection], spec.min_gap):
                continue
            if _clashes(candidate, placed, spec.min_gap) or _clashes(
                reflection, placed, spec.min_gap
            ):
                continue
            placed.extend((candidate, reflection))
            break
        else:
            return None

    return placed


def _sample_centre_piece(
    spec: RandomTerrainConfig,
    board: BoardDimensions,
    rng: Generator,
) -> Polygon:
    """Draw the piece that straddles the centre line of a mirrored layout.

    Built symmetric by construction rather than by hulling a shape with its own
    reflection: half the vertices are sampled on the right of the axis and mirrored to
    the left. That fixes the vertex count at twice the half-count, which keeps it
    inside the observation's vertex budget — hulling could produce twice as many
    vertices as the piece was drawn with, and the extra ones would be truncated away.
    """
    width = float(rng.uniform(spec.min_size, spec.max_size))
    height = float(rng.uniform(spec.min_size, spec.max_size))
    centre_x = board.width / 2.0
    centre_y = float(
        rng.uniform(
            spec.edge_margin + height / 2.0,
            board.height - spec.edge_margin - height / 2.0,
        )
    )

    half_count = int(rng.integers(_MIN_VERTICES // 2, _MAX_VERTICES // 2 + 1))
    # Angles strictly inside the right half-plane, in order from bottom to top, so the
    # mirrored half closes the outline without duplicating a vertex on the axis.
    slice_width = np.pi / (half_count + 1)
    angles = -np.pi / 2.0 + slice_width * (np.arange(half_count) + 1)
    radii = rng.uniform(_MIN_RADIUS_FRACTION, 1.0, size=half_count)

    right = np.column_stack(
        [
            centre_x + radii * np.cos(angles) * width / 2.0,
            centre_y + radii * np.sin(angles) * height / 2.0,
        ]
    )
    left = right[::-1].copy()
    left[:, 0] = board.width - left[:, 0]
    return Polygon(np.vstack([right, left]))


def generate_terrain(
    spec: RandomTerrainConfig,
    board: BoardDimensions,
    rng: Generator,
    blocking_mask: list[list[bool]] | None = None,
) -> Terrain:
    """Generate a random, optionally mirrored terrain layout.

    The number of footprints always equals ``spec.count`` — observation batching
    stacks terrain into one array, so the count must not vary between episodes.

    Raises:
        RuntimeError: if no valid layout was found. The config validator rules
            out over-packed specs, so this means the sampler was unlucky enough
            to fail every attempt.
    """
    for _ in range(_MAX_LAYOUT_ATTEMPTS):
        placed = _attempt_layout(spec, board, rng)
        if placed is not None and len(placed) == spec.count:
            return Terrain(
                [Footprint(polygon) for polygon in placed],
                blocking_mask=blocking_mask,
            )

    raise RuntimeError(
        f"could not place {spec.count} terrain pieces of up to "
        f"{spec.max_size}x{spec.max_size} on a {board.width}x{board.height} board "
        f"in {_MAX_LAYOUT_ATTEMPTS} attempts"
    )
