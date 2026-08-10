"""Terrain placement domain service: generate random ruin layouts per episode.

Mirrors `placement.py` in shape — a pure function taking the config, the board
and an rng, returning a domain object. Placement is rejection sampling; the
config validator rejects layouts that pack too tightly for it to converge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from wargame_rl.wargame.envs.domain.terrain import Footprint, Terrain
from wargame_rl.wargame.envs.domain.value_objects import BoardDimensions
from wargame_rl.wargame.envs.types.config import RandomTerrainConfig

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
    """Reflect a rectangle across the board's vertical centre line."""
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
            return Terrain([Footprint.from_corners(*rect) for rect in placed])

    raise RuntimeError(
        f"could not place {spec.count} terrain pieces of up to "
        f"{spec.max_size}x{spec.max_size} on a {board.width}x{board.height} board "
        f"in {_MAX_LAYOUT_ATTEMPTS} attempts"
    )
