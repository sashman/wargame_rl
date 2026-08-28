"""The sampling grid every board-wide read shares.

One grid, several readers. The sight shadow, the threat overlay and the threat
field all answer questions about *the same piece of ground*, and the moment two
of them sample different points their answers stop being comparable — a cell one
calls safe and another calls hidden is a disagreement about the sampler, not
about the board, and it looks exactly like a real finding.

The arithmetic here is `renders/v2/control.py::_cell_centres` verbatim, promoted
out of the renderer so the analysis layer can share it. That direction matters:
`board/` is a leaf, so a heatmap that needed the renderer's private helper would
have had to copy it, which is how the two would drift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from wargame_rl.wargame.envs.domain.battle_view import BattleView

# Matches `renders/v2/control.py`'s SHADOW_SPACING and THREAT_SPACING. Changing
# it here would silently decouple the field from the overlays it is read beside.
DEFAULT_SPACING = 1.0


@dataclass(frozen=True, slots=True)
class BoardGrid:
    """A regular sampling of the board, in board units.

    `centres` is row-major (`indexing="xy"` over rows then columns), which is
    what makes `as_image` a plain reshape.
    """

    centres: np.ndarray
    n_rows: int
    n_cols: int
    spacing: float
    width: float
    height: float

    @property
    def n_cells(self) -> int:
        """How many cells the board was sampled at."""
        return self.n_rows * self.n_cols

    def as_image(self, flat: np.ndarray) -> np.ndarray:
        """Reshape a per-cell `(Q,)` array into `(n_rows, n_cols)` for drawing."""
        if flat.shape[0] != self.n_cells:
            raise ValueError(f"expected {self.n_cells} cells, got {flat.shape[0]}")
        return flat.reshape(self.n_rows, self.n_cols)

    def nearest(self, points: np.ndarray) -> np.ndarray:
        """`(N,)` index of the cell each `(N, 2)` point falls in.

        Computed from the grid's own definition rather than by searching
        `centres`, so it costs nothing at 25 models and stays exact at the
        clamped edge cells, whose centres are not on the regular lattice.
        """
        if points.size == 0:
            return np.zeros(0, dtype=np.intp)
        columns = np.clip(
            np.floor(points[:, 0] / self.spacing).astype(np.intp), 0, self.n_cols - 1
        )
        rows = np.clip(
            np.floor(points[:, 1] / self.spacing).astype(np.intp), 0, self.n_rows - 1
        )
        indices: np.ndarray = rows * self.n_cols + columns
        return indices


def board_grid(
    width: float, height: float, spacing: float = DEFAULT_SPACING
) -> BoardGrid:
    """Cell centres covering a `width` x `height` board at `spacing`.

    Centres are clamped inside the board, so a partial edge cell is still
    sampled somewhere it exists rather than off the table.

    Takes plain floats rather than a view so a test needs no environment.
    """
    if spacing <= 0:
        raise ValueError(f"spacing must be positive, got {spacing}")
    board_w = float(width)
    board_h = float(height)
    n_cols = max(1, math.ceil(board_w / spacing))
    n_rows = max(1, math.ceil(board_h / spacing))
    xs = np.minimum((np.arange(n_cols) + 0.5) * spacing, board_w)
    ys = np.minimum((np.arange(n_rows) + 0.5) * spacing, board_h)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    return BoardGrid(
        centres=np.column_stack([grid_x.ravel(), grid_y.ravel()]),
        n_rows=n_rows,
        n_cols=n_cols,
        spacing=float(spacing),
        width=board_w,
        height=board_h,
    )


def board_grid_for(view: BattleView, spacing: float = DEFAULT_SPACING) -> BoardGrid:
    """`board_grid` sized from a live board."""
    return board_grid(
        float(view.config.board_width), float(view.config.board_height), spacing
    )
