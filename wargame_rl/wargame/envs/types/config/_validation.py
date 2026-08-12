"""Shared validation helpers for the config models.

Private to `types.config`: these are the checks the models reuse, kept out of
the model modules so that none of them has to import another for a helper.
"""

from __future__ import annotations

from typing import Protocol, TypeVar


class _HasCoords(Protocol):
    x: float | None
    y: float | None


_CoordsT = TypeVar("_CoordsT", bound=_HasCoords)


def _validate_coords_both_or_neither(x: float | None, y: float | None) -> None:
    """Raise if exactly one of x, y is None."""
    if (x is None) != (y is None):
        raise ValueError("x and y must both be set or both be None")


def _validate_entity_configs(
    count: int,
    configs: list[_CoordsT] | None,
    board_width: int,
    board_height: int,
    entity_name: str,
) -> None:
    """Validate entity list length, all-or-none coords, and in-bounds for fixed positions."""
    if configs is None:
        return
    if len(configs) != count:
        raise ValueError(
            f"{entity_name} has {len(configs)} entries but expected {count}"
        )
    has_coords = [c.x is not None for c in configs]
    if any(has_coords) and not all(has_coords):
        raise ValueError(f"Either all {entity_name} must have x/y coordinates or none")
    for i, c in enumerate(configs):
        if (
            c.x is not None
            and c.y is not None
            and (c.x >= board_width or c.y >= board_height)
        ):
            raise ValueError(
                f"{entity_name}[{i}] ({c.x}, {c.y}) is outside "
                f"the board ({board_width}x{board_height})"
            )


def _normalise_rect(r: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = r
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


# Rejection sampling for random terrain slows sharply as the board fills, so
# layouts are rejected at config load well before that point.
_MAX_TERRAIN_PACKING_FRACTION = 0.5


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 <= bx1 and bx0 <= ax1 and ay0 <= by1 and by0 <= ay1
