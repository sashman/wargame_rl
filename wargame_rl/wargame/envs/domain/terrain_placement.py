"""Terrain placement domain service: generate random ruin layouts per episode.

Mirrors `placement.py` in shape — a pure function taking the config, the board
and an rng, returning a domain object. Placement is rejection sampling; the
config validator rejects layouts that pack too tightly for it to converge.

Pieces are sampled as *cell rectangles* and converted at the boundary, or as
convex n-gons inscribed in that rectangle when `n_vertices` is set. The
rectangle stays the sampling primitive either way, because the packing
constraints (edge margin, minimum gap, mirroring) are all easier to reason about
on a box and an inscribed outline can only be smaller.
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

_Rect = tuple[int, int, int, int]


def _clashes(candidate: _Rect, placed: list[_Rect], min_gap: int) -> bool:
    """True if `candidate` is within `min_gap` cells of anything already placed."""
    x0, y0, x1, y1 = candidate
    for px0, py0, px1, py1 in placed:
        if (
            x0 - min_gap <= px1
            and px0 - min_gap <= x1
            and y0 - min_gap <= py1
            and py0 - min_gap <= y1
        ):
            return True
    return False


def _mirror(rect: _Rect, board_width: int) -> _Rect:
    """Reflect a cell rectangle across the board's vertical centre line.

    In *cell index* space, so `board_width - 1 - x`: cell `i` pairs with cell
    `W - 1 - i`. That is consistent with reflecting the continuous rectangle
    about `board_width`, because `from_cell_rect` pushes the far corner out by
    one on both — get one of the two wrong and the layout stops being symmetric.
    """
    x0, y0, x1, y1 = rect
    return (board_width - 1 - x1, y0, board_width - 1 - x0, y1)


def _sample_piece(
    spec: RandomTerrainConfig,
    board: BoardDimensions,
    rng: Generator,
) -> _Rect:
    """Draw one footprint uniformly from the region inside the edge margin."""
    width = int(rng.integers(spec.min_size, spec.max_size + 1))
    height = int(rng.integers(spec.min_size, spec.max_size + 1))
    x0 = int(rng.integers(spec.edge_margin, board.width - spec.edge_margin - width + 1))
    y0 = int(
        rng.integers(spec.edge_margin, board.height - spec.edge_margin - height + 1)
    )
    return (x0, y0, x0 + width - 1, y0 + height - 1)


def _sample_centre_piece(
    spec: RandomTerrainConfig,
    board: BoardDimensions,
    rng: Generator,
) -> _Rect:
    """Draw the piece that straddles the centre line of a mirrored layout.

    An odd piece count leaves one piece without a partner, so it has to be its
    own mirror image. That needs `board_width - width` to be even; the width is
    nudged by one within the configured range when it is not.
    """
    width = int(rng.integers(spec.min_size, spec.max_size + 1))
    if (board.width - width) % 2 != 0:
        width = width + 1 if width < spec.max_size else max(spec.min_size, width - 1)
    height = int(rng.integers(spec.min_size, spec.max_size + 1))
    x0 = (board.width - width) // 2
    y0 = int(
        rng.integers(spec.edge_margin, board.height - spec.edge_margin - height + 1)
    )
    return (x0, y0, x0 + width - 1, y0 + height - 1)


def _inscribed_polygon(rect: _Rect, n_vertices: int, rng: Generator) -> Polygon:
    """A convex n-gon inscribed in the cell rectangle's continuous extent.

    Vertices are drawn at sorted random angles on the rectangle's inscribed
    ellipse, so the outline is convex by construction and the angles are never
    duplicated — a repeated angle would give two coincident vertices and a
    degenerate edge.
    """
    x0, y0, x1, y1 = rect
    left, bottom = float(x0), float(y0)
    right, top = float(x1) + 1.0, float(y1) + 1.0
    centre_x, centre_y = (left + right) / 2.0, (bottom + top) / 2.0
    radius_x, radius_y = (right - left) / 2.0, (top - bottom) / 2.0

    # One angle per sector, jittered within it: this both spreads the vertices
    # and guarantees a strict ordering, which a plain sort of uniform draws does
    # not (two equal angles would collapse an edge).
    sector = 2.0 * np.pi / n_vertices
    angles = sector * (np.arange(n_vertices) + rng.uniform(0.15, 0.85, n_vertices))
    return Polygon(
        np.column_stack(
            [centre_x + radius_x * np.cos(angles), centre_y + radius_y * np.sin(angles)]
        )
    )


def _to_footprint(rect: _Rect, spec: RandomTerrainConfig, rng: Generator) -> Footprint:
    """Turn a sampled cell rectangle into the piece the domain holds."""
    if spec.n_vertices is None:
        return Footprint.from_cell_rect(*rect)
    return Footprint(_inscribed_polygon(rect, spec.n_vertices, rng))


def _attempt_layout(
    spec: RandomTerrainConfig,
    board: BoardDimensions,
    rng: Generator,
) -> list[_Rect] | None:
    """Build one candidate layout, or None if a piece could not be placed."""
    placed: list[_Rect] = []

    if spec.mirror and spec.count % 2 == 1:
        placed.append(_sample_centre_piece(spec, board, rng))

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
            reflection = _mirror(candidate, board.width)
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


def _build_footprints(
    placed: list[_Rect],
    spec: RandomTerrainConfig,
    board: BoardDimensions,
    rng: Generator,
) -> list[Footprint]:
    """Resolve placed rectangles into footprints, preserving mirror symmetry.

    A mirrored *pair* has to be one outline and its reflection, not two
    independent draws inscribed in reflected boxes — inscribing twice would give
    two different shapes on ground that is supposed to be equal, which is the
    exact bias `mirror` exists to remove. The unpaired centre piece of an odd
    layout is built symmetric by construction for the same reason: sample half
    its vertices and mirror them, rather than hulling a shape with its own
    reflection, which doubles the vertex count and overflows the budget.
    """
    if spec.n_vertices is None:
        return [Footprint.from_cell_rect(*rect) for rect in placed]

    footprints: list[Footprint] = []
    index = 0
    if spec.mirror and spec.count % 2 == 1:
        footprints.append(
            Footprint(_symmetric_polygon(placed[0], spec.n_vertices, rng))
        )
        index = 1

    while index < len(placed):
        polygon = _inscribed_polygon(placed[index], spec.n_vertices, rng)
        footprints.append(Footprint(polygon))
        if spec.mirror:
            footprints.append(Footprint(polygon.mirrored(float(board.width))))
            index += 2
        else:
            index += 1
    return footprints


def _symmetric_polygon(rect: _Rect, n_vertices: int, rng: Generator) -> Polygon:
    """An outline that is its own mirror image about its vertical centre line.

    Built by drawing *half* the vertices on the right of the axis and reflecting
    them, so symmetry is exact and the vertex count is exactly the budget. The
    obvious alternative — hull a random shape with its own reflection — doubles
    the vertex count, which then overflows the observation's outline budget.

    An odd budget leaves one vertex over; it goes on the axis itself, at the top,
    which is the only place it can sit without breaking the symmetry.
    """
    x0, y0, x1, y1 = rect
    left, bottom = float(x0), float(y0)
    right, top = float(x1) + 1.0, float(y1) + 1.0
    centre_x, centre_y = (left + right) / 2.0, (bottom + top) / 2.0
    radius_x, radius_y = (right - left) / 2.0, (top - bottom) / 2.0

    half = n_vertices // 2
    sector = np.pi / half
    # Bottom to top through the right half of the inscribed ellipse.
    angles = -np.pi / 2.0 + sector * (np.arange(half) + rng.uniform(0.15, 0.85, half))
    right_x = centre_x + radius_x * np.cos(angles)
    axis_y = centre_y + radius_y * np.sin(angles)

    vertices = [(float(x), float(y)) for x, y in zip(right_x, axis_y)]
    if n_vertices % 2 == 1:
        vertices.append((centre_x, centre_y + radius_y))
    # ...and back down the left half, which keeps the winding anticlockwise.
    vertices.extend(
        (float(2.0 * centre_x - x), float(y))
        for x, y in zip(right_x[::-1], axis_y[::-1])
    )
    return Polygon.from_points(vertices)


def generate_terrain(
    spec: RandomTerrainConfig,
    board: BoardDimensions,
    rng: Generator,
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
            return Terrain(_build_footprints(placed, spec, board, rng))

    raise RuntimeError(
        f"could not place {spec.count} terrain pieces of up to "
        f"{spec.max_size}x{spec.max_size} on a {board.width}x{board.height} board "
        f"in {_MAX_LAYOUT_ATTEMPTS} attempts"
    )
