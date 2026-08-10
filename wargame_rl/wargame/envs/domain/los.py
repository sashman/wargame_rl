"""Grid line-of-sight: integer Bresenham ray and injectable blocking predicate.

Coordinates match the env: x spans ``0 .. width-1``, y spans ``0 .. height-1``.
``is_blocking(x, y)`` is evaluated only for **strictly interior** cells along the
segment (endpoints excluded), per phase context.

Algorithm: standard 2D Bresenham line (error-term loop), inclusive of both endpoints
in ``iter_los_cells``; see e.g. Wikipedia "Bresenham's line algorithm".
"""

from __future__ import annotations

from collections.abc import Callable


def _bresenham_line(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Return all grid cells on the line from (x0,y0) to (x1,y1), inclusive."""
    points: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return points


def _in_bounds(x: int, y: int, width: int, height: int) -> bool:
    return 0 <= x < width and 0 <= y < height


def iter_los_cells(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    """Cells along the Bresenham segment, inclusive. Empty if an endpoint is OOB."""
    if not _in_bounds(x0, y0, width, height):
        return []
    if not _in_bounds(x1, y1, width, height):
        return []
    return _bresenham_line(x0, y0, x1, y1)


def has_line_of_sight(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    width: int,
    height: int,
    is_blocking: Callable[[int, int], bool],
) -> bool:
    """True if every strictly-interior cell on the LOS segment is non-blocking."""
    cells = iter_los_cells(x0, y0, x1, y1, width, height)
    if not cells:
        return False
    if len(cells) == 1:
        return True
    for x, y in cells[1:-1]:
        if is_blocking(x, y):
            return False
    return True
