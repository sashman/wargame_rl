"""Domain reads the v2 renderer needs, returned as plain data.

The legacy renderer computed objective ownership and the debug LOS verdict inside
its draw methods, mixing domain calls into drawing. v2 keeps drawing
(`scene`/`backend`) domain-free by doing those reads here and handing the results
to `build_scene` as data. This is the one place v2 touches the domain, and it
reuses the exact functions the legacy renderer used so the results match.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from wargame_rl.wargame.envs.domain.battle_view import BattleView
from wargame_rl.wargame.envs.domain.entities import alive_mask_for
from wargame_rl.wargame.envs.env_components.distance_cache import (
    compute_distances,
    objective_ownership_from_norms_offset,
)
from wargame_rl.wargame.envs.renders.v2.scene import Control


def compute_objective_control(view: BattleView) -> tuple[Control, ...]:
    """Ownership per objective, mirroring the legacy ``_draw_target`` body.

    Reproduced exactly, including the empty-opponent branch that feeds a
    ``(0, n_obj)`` norms array so a board with no opponents still resolves.
    """
    objectives = view.objectives
    if not objectives:
        return ()

    player_models = view.player_models
    opponent_models = view.opponent_models
    n_obj = len(objectives)

    player_alive = alive_mask_for(player_models)
    player_cache = compute_distances(player_models, objectives, alive_mask=player_alive)
    if opponent_models:
        opponent_alive = alive_mask_for(opponent_models)
        opponent_cache = compute_distances(
            opponent_models, objectives, alive_mask=opponent_alive
        )
        opponent_norms = opponent_cache.model_obj_norms_offset
    else:
        opponent_norms = np.zeros((0, n_obj), dtype=np.float64)

    player_controls, opponent_controls = objective_ownership_from_norms_offset(
        player_cache.model_obj_norms_offset,
        opponent_norms,
        player_cache.obj_radii,
    )

    result: list[Control] = []
    for i in range(n_obj):
        if player_controls[i]:
            result.append(Control.PLAYER)
        elif opponent_controls[i]:
            result.append(Control.OPPONENT)
        else:
            result.append(Control.NEUTRAL)
    return tuple(result)


@dataclass(frozen=True)
class LosResult:
    """The debug sight line and whether it is clear."""

    clear: bool
    a: tuple[float, float]
    b: tuple[float, float]


def probe_debug_los(view: BattleView) -> LosResult | None:
    """First alive player to first alive opponent; ``None`` if either is absent."""
    player_alive = alive_mask_for(view.player_models)
    p_idx = next((i for i, ok in enumerate(player_alive) if ok), None)
    if p_idx is None or not view.opponent_models:
        return None
    opponent_alive = alive_mask_for(view.opponent_models)
    o_idx = next((i for i, ok in enumerate(opponent_alive) if ok), None)
    if o_idx is None:
        return None

    pm = view.player_models[p_idx]
    om = view.opponent_models[o_idx]
    a = (float(pm.location[0]), float(pm.location[1]))
    b = (float(om.location[0]), float(om.location[1]))
    clear = view.has_line_of_sight_between_points(a[0], a[1], b[0], b[1])
    return LosResult(clear=clear, a=a, b=b)


# One sample per square inch. The board is 60x44, so a full sweep is 2640 rays
# in a single vectorised call -- and a finer grid buys detail the eye cannot use
# on a shape whose edges are already only as accurate as the sample spacing.
SHADOW_SPACING = 1.0

# `(x0, y0, x1, y1)` in board units, half-open in both axes.
ShadowRect = tuple[float, float, float, float]


def sight_from(
    view: BattleView,
    origin: tuple[float, float],
    targets: np.ndarray,
) -> np.ndarray:
    """``(Q,)`` — whether `origin` can see each target point.

    `BattleView.line_of_sight_matrix` is the predicate both shooting masks and
    the exposure scan use, so the shading is a picture of the same question the
    game asks. It is not the shooting *mask*, which also gates on weapon range.
    """
    return np.asarray(
        view.line_of_sight_matrix(np.array([origin], dtype=float), targets)[0]
    )


def compute_los_shadow(
    view: BattleView,
    origin: tuple[float, float],
    *,
    spacing: float = SHADOW_SPACING,
) -> tuple[ShadowRect, ...]:
    """Where `origin` cannot see, as merged rectangles in board units.

    **The engine's own predicate is sampled rather than the shadow geometry being
    computed.** Projecting terrain silhouettes would be cheaper and would give
    exact edges, but it would be the *renderer's* answer to "what is hidden", and
    the whole reason to draw this is to see the answer sight resolution actually
    gives — a disagreement between the two is the bug being hunted, and a
    renderer that computed its own would hide it.
    """
    board_w = float(view.config.board_width)
    board_h = float(view.config.board_height)
    n_cols = max(1, math.ceil(board_w / spacing))
    n_rows = max(1, math.ceil(board_h / spacing))

    # Cell *centres*, clamped inside the board so a partial edge cell is still
    # sampled somewhere it exists rather than off the table.
    xs = np.minimum((np.arange(n_cols) + 0.5) * spacing, board_w)
    ys = np.minimum((np.arange(n_rows) + 0.5) * spacing, board_h)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    targets = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    visible = sight_from(view, origin, targets).reshape(n_rows, n_cols)
    return _merge_hidden(~visible, spacing, board_w, board_h)


def _merge_hidden(
    hidden: np.ndarray, spacing: float, board_w: float, board_h: float
) -> tuple[ShadowRect, ...]:
    """Merge a ``[row][col]`` hidden mask into rectangles, rows first then down.

    Both directions, not just rows: the fills are translucent, so two rectangles
    meeting on an edge double the alpha along that seam and a shadow drawn as one
    strip per row comes out combed. Merging vertically also cuts a large hidden
    region from dozens of primitives to one.
    """
    n_rows, n_cols = hidden.shape
    rects: list[ShadowRect] = []
    # Column span -> the row it started on, for spans still growing downward.
    open_runs: dict[tuple[int, int], int] = {}

    def close(span: tuple[int, int], row_from: int, row_to: int) -> None:
        rects.append(
            (
                span[0] * spacing,
                row_from * spacing,
                min(span[1] * spacing, board_w),
                min(row_to * spacing, board_h),
            )
        )

    for row in range(n_rows + 1):
        spans = _row_spans(hidden[row]) if row < n_rows else set()
        for span, started in list(open_runs.items()):
            if span not in spans:
                close(span, started, row)
                del open_runs[span]
        for span in spans:
            open_runs.setdefault(span, row)
    return tuple(rects)


def _row_spans(row: np.ndarray) -> set[tuple[int, int]]:
    """Half-open ``(start, end)`` column spans of the True runs in one row."""
    padded = np.concatenate(([False], row.astype(bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return {(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)}
