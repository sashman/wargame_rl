"""The promoted grid is the renderer's old private one, cell for cell.

`board_grid` was `renders/v2/control.py::_cell_centres`. Two shipped overlays
are pixel-pinned to that arithmetic (`test_threat_overlay`, `test_debug_session`),
so a promotion that changed a single centre would move drawings that are asserted
elsewhere -- and the failure would surface as an unrelated overlay test, not as
this one. Pinning it here says what actually broke.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from wargame_rl.wargame.envs.board.grid import board_grid

# (width, height, spacing): the shipped board, both overlay spacings, an odd
# board, and a spacing coarser than the board -- the case where `max(1, ...)`
# is the only thing standing between the caller and an empty grid.
CASES = [(60, 44, 1.0), (60, 44, 2.0), (60, 44, 0.5), (37, 23, 1.0), (5, 5, 7.0)]


def _original(
    width: float, height: float, spacing: float
) -> tuple[np.ndarray, int, int]:
    """`_cell_centres`'s body as it stood before the promotion."""
    n_cols = max(1, math.ceil(width / spacing))
    n_rows = max(1, math.ceil(height / spacing))
    xs = np.minimum((np.arange(n_cols) + 0.5) * spacing, width)
    ys = np.minimum((np.arange(n_rows) + 0.5) * spacing, height)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    return np.column_stack([grid_x.ravel(), grid_y.ravel()]), n_rows, n_cols


@pytest.mark.parametrize(("width", "height", "spacing"), CASES)
def test_the_promoted_grid_matches_the_renderers_original(
    width: float, height: float, spacing: float
) -> None:
    """Arrange a board, act by sampling it both ways, assert they agree exactly."""
    centres, n_rows, n_cols = _original(width, height, spacing)

    grid = board_grid(width, height, spacing)

    assert np.array_equal(grid.centres, centres)
    assert (grid.n_rows, grid.n_cols) == (n_rows, n_cols)


@pytest.mark.parametrize(("width", "height", "spacing"), CASES)
def test_every_centre_lands_on_the_board(
    width: float, height: float, spacing: float
) -> None:
    """A partial edge cell is sampled somewhere it exists, not off the table."""
    grid = board_grid(width, height, spacing)

    assert (grid.centres >= 0.0).all()
    assert (grid.centres[:, 0] <= width).all()
    assert (grid.centres[:, 1] <= height).all()


def test_nearest_returns_the_cell_a_point_falls_in() -> None:
    """`nearest` is derived from the lattice, not searched over centres.

    The edge cells matter: their centres are *clamped* onto the board edge and
    so are not on the regular lattice, which is exactly where a nearest-centre
    search and a floor-divide disagree.
    """
    grid = board_grid(60, 44, 1.0)
    points = np.array([[0.0, 0.0], [0.5, 0.5], [59.99, 43.99], [30.2, 20.7]])

    cells = grid.nearest(points)

    assert cells[0] == cells[1] == 0
    assert cells[2] == grid.n_cells - 1
    assert cells[3] == 20 * grid.n_cols + 30


def test_a_reshaped_field_is_row_major() -> None:
    """`as_image` is a plain reshape, so row 0 is the low-y edge of the board."""
    grid = board_grid(4, 3, 1.0)
    flat = np.arange(grid.n_cells, dtype=float)

    image = grid.as_image(flat)

    assert image.shape == (3, 4)
    assert image[1, 2] == flat[1 * 4 + 2]


def test_a_wrong_sized_field_is_refused() -> None:
    """Reshaping the wrong length silently would misplace every cell."""
    grid = board_grid(4, 3, 1.0)

    with pytest.raises(ValueError, match="expected 12 cells"):
        grid.as_image(np.zeros(11))


def test_a_non_positive_spacing_is_refused_at_construction() -> None:
    """Validation at construction, not several thousand frames later."""
    with pytest.raises(ValueError, match="spacing must be positive"):
        board_grid(60, 44, 0.0)
