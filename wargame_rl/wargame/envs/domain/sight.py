"""Whether one cell can see another: what blocks sight, composed with how it is traced.

This is deliberately separate from `los.py`. That module is the geometry
primitive — a Bresenham ray and an injectable predicate — and knows nothing
about terrain or the game's blocking rules. This module is the *answer*, built
by handing that primitive the two things that actually block a shot: the static
`blocking_mask` from config, and the terrain footprints that do not contain
either endpoint.

Keeping the composition here rather than on the env facade gives the question
"can A see B?" a single home in the domain. It is also the seam that changes
most when the board stops being a grid: continuous coordinates, a three-state
answer for cover, and endpoints that carry a base radius all land on these two
functions rather than on every caller.

Terrain is read at call time, never cached. `Battle.set_terrain` replaces the
layout between episodes and a cache here would need invalidating; the footprint
scan is cheap next to the ray.
"""

from __future__ import annotations

from collections.abc import Callable

from wargame_rl.wargame.envs.domain.los import has_line_of_sight
from wargame_rl.wargame.envs.domain.terrain import Terrain

BlockingMask = list[list[bool]]


def blocking_predicate_for_query(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    terrain: Terrain,
    blocking_mask: BlockingMask | None,
) -> Callable[[int, int], bool]:
    """Build the per-query predicate: is this cell opaque for *this* sight line?

    Terrain is filtered per query rather than globally, because a piece
    containing either endpoint does not block — a model can see out of the ruin
    it stands in, and can be seen when standing in one (the see-out / see-into
    rule, `docs/rules/13-terrain.md`). The static `blocking_mask` is opaque
    matter and gets no such exemption.
    """
    active = terrain.blocking_footprints_for_endpoints(x0, y0, x1, y1)

    def is_blocking(x: int, y: int) -> bool:
        if blocking_mask is not None and blocking_mask[y][x]:
            return True
        return any(footprint.contains(x, y) for footprint in active)

    return is_blocking


def has_line_of_sight_between_cells(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    board_width: int,
    board_height: int,
    terrain: Terrain,
    blocking_mask: BlockingMask | None,
) -> bool:
    """True when sight is clear between two cells.

    Endpoints are sorted into a canonical order first, which makes the answer
    exactly symmetric: A sees B if and only if B sees A. Several metrics depend
    on that — `firepower_ratio` reads an exposed model as one that can also
    fire — and a Bresenham ray is not otherwise guaranteed to trace the same
    cells in both directions.
    """
    (ax, ay), (bx, by) = sorted([(x0, y0), (x1, y1)])
    return has_line_of_sight(
        ax,
        ay,
        bx,
        by,
        board_width,
        board_height,
        blocking_predicate_for_query(ax, ay, bx, by, terrain, blocking_mask),
    )
