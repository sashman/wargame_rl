"""Domain reads the v2 renderer needs, returned as plain data.

The legacy renderer computed objective ownership and the debug LOS verdict inside
its draw methods, mixing domain calls into drawing. v2 keeps drawing
(`scene`/`backend`) domain-free by doing those reads here and handing the results
to `build_scene` as data. This is the one place v2 touches the domain, and it
reuses the exact functions the legacy renderer used so the results match.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from wargame_rl.wargame.envs.domain.battle_view import BattleView
from wargame_rl.wargame.envs.domain.entities import WargameModel, alive_mask_for
from wargame_rl.wargame.envs.domain.sight import BlockingMask, line_of_sight_matrix
from wargame_rl.wargame.envs.domain.terrain import Terrain
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
    targets, n_rows, n_cols = _cell_centres(view, spacing)
    visible = sight_from(view, origin, targets).reshape(n_rows, n_cols)
    return _merge_hidden(~visible, spacing, board_w, board_h)


def _cell_centres(view: BattleView, spacing: float) -> tuple[np.ndarray, int, int]:
    """``(Q, 2)`` cell centres covering the board, plus the grid shape.

    Shared by the sight shadow and the threat sweep so the two can never sample
    different points and disagree about the same piece of ground.

    Centres are clamped inside the board, so a partial edge cell is still
    sampled somewhere it exists rather than off the table.
    """
    board_w = float(view.config.board_width)
    board_h = float(view.config.board_height)
    n_cols = max(1, math.ceil(board_w / spacing))
    n_rows = max(1, math.ceil(board_h / spacing))
    xs = np.minimum((np.arange(n_cols) + 0.5) * spacing, board_w)
    ys = np.minimum((np.arange(n_rows) + 0.5) * spacing, board_h)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    return np.column_stack([grid_x.ravel(), grid_y.ravel()]), n_rows, n_cols


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


# --- threat overlays --------------------------------------------------------

# Matches SHADOW_SPACING. Range-gated, a full 25-origin sweep at this spacing
# costs ~66-105 ms per side against ~116 ms for the single-observer shadow that
# already ships; ungated the same sweep is 3.0 s, so the gate is not optional.
THREAT_SPACING = 1.0

# Chaikin iterations. The displacement does not grow with iterations -- measured,
# it converges to exactly an eighth of a cell (0.063 / 0.094 / 0.109 / 0.117 /
# 0.121 at 1..6 on a 1" grid) -- so smoothing harder costs vertices and never
# fidelity. Three takes 75% of the available smoothing at a quarter of the vertex
# count of five. It is bounded at all only because the rings keep every unit-cell
# vertex; see `_rings_from_mask`.
THREAT_SMOOTHING = 3

# A closed ring of board-unit vertices.
Ring = tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class ThreatOptions:
    """How the threat overlays are drawn, and whether they are drawn at all.

    A renderer setting, deliberately **not** a `WargameEnvConfig` field: every
    config model is `extra="forbid"` and enumerated by
    `test_config_rejects_unknown_keys`, and the config is the *scenario*. A
    purely visual switch living there would change what a video looks like for a
    given config across every run that ever used it.

    Validated here rather than at each draw, so an impossible spacing fails when
    the renderer is built and not several thousand frames later.
    """

    show_threat: bool = False
    show_engagement: bool = False
    spacing: float = THREAT_SPACING
    smoothing: int = THREAT_SMOOTHING

    def __post_init__(self) -> None:
        if self.spacing <= 0:
            raise ValueError(f"spacing must be positive, got {self.spacing}")
        if self.smoothing < 0:
            raise ValueError(f"smoothing must be >= 0, got {self.smoothing}")

    @property
    def enabled(self) -> bool:
        """Whether anything would be drawn, and so whether to sweep at all."""
        return self.show_threat or self.show_engagement

    def toggled(
        self, *, threat: bool | None = None, engagement: bool | None = None
    ) -> "ThreatOptions":
        """A copy with one or both switches flipped — for the runtime keys."""
        return replace(
            self,
            show_threat=self.show_threat if threat is None else threat,
            show_engagement=(
                self.show_engagement if engagement is None else engagement
            ),
        )


@dataclass(frozen=True)
class ThreatOverlay:
    """Both sides' threat geometry for one frame, in board units.

    Geometry only. Colour belongs to the theme and is applied in `build_scene`,
    which is the module that owns the palette.
    """

    player_engagement: tuple[tuple[float, float], ...] = ()
    opponent_engagement: tuple[tuple[float, float], ...] = ()
    engagement_radius: float = 0.0
    player_threat: tuple[Ring, ...] = ()
    opponent_threat: tuple[Ring, ...] = ()

    def is_empty(self) -> bool:
        """Whether there is nothing to draw."""
        return not (
            self.player_engagement
            or self.opponent_engagement
            or self.player_threat
            or self.opponent_threat
        )


def engagement_radius(view: BattleView) -> float:
    """Distance from a model's centre at which it engages an enemy centre.

    The gate is `nearest_live - 2 * base_radius > engagement_range`
    (`env_components/shooting_masks.py`), so the radius is
    `engagement_range + 2 * base_radius`.

    **The global `rules_quantities.base_radius`, not `model.base_radius`.** The
    engine measures the gate with the config's resolved global; drawing it from
    the per-model value would put the picture and the rule in silent
    disagreement the day a config sets them differently.
    """
    rules = view.rules_quantities
    return float(rules.engagement_range + 2.0 * rules.base_radius)


def compute_engagement_zone(
    models: list[WargameModel],
) -> tuple[tuple[float, float], ...]:
    """Centres of the alive models that project an engagement zone.

    Casualties are excluded because the engine excludes them — the gate masks
    dead targets to infinity, which is the corpse bug that pinned models for a
    whole episode until 2026-08-19. A picture that kept drawing them would be
    drawing the bug.
    """
    return tuple(
        (float(model.location[0]), float(model.location[1]))
        for model, alive in zip(models, alive_mask_for(models), strict=True)
        if alive
    )


def compute_threat_region(
    view: BattleView,
    models: list[WargameModel],
    max_ranges: np.ndarray,
    *,
    spacing: float = THREAT_SPACING,
    smooth: int = THREAT_SMOOTHING,
) -> tuple[Ring, ...]:
    """Ground a side can shoot at, as closed rings in board units.

    Range **and** line of sight, sampled on a grid: a cell is threatened when
    some alive model of this side is within its own weapon range of it and can
    see it. Sight is `BattleView.line_of_sight_matrix`, the predicate the
    shooting mask itself uses, so the drawing cannot disagree with the rule.

    Deliberately **not** gated on `advanced_this_turn` or on whether the shooter
    is itself engaged. Both appear in `compute_shooting_masks` and both are
    excluded from `compute_threat_counts` for the reason its docstring gives:
    they describe whose turn it is rather than who is dangerous, and a model in
    base contact is still a threat next round. This draws danger.
    """
    targets, n_rows, n_cols = _cell_centres(view, spacing)
    alive = alive_mask_for(models)
    if not alive.any():
        return ()
    origins = np.array(
        [[float(m.location[0]), float(m.location[1])] for m in models], dtype=float
    )[alive]
    ranges = np.asarray(max_ranges, dtype=float)[alive]

    distances = np.linalg.norm(origins[:, None, :] - targets[None, :, :], axis=2)
    # `ranges > 0` is load-bearing: an unarmed model has range 0.0, and `0 <= 0`
    # would mark the cell it is standing on as threatened by a model that cannot
    # shoot at all. Same guard `compute_threat_counts` documents.
    candidates = (distances <= ranges[:, None]) & (ranges > 0)[:, None]
    if not candidates.any():
        return ()

    # The mask is what makes this affordable -- `domain/sight.py` builds its pair
    # list from `np.nonzero(candidates)`, so a ruled-out pair is never traced.
    visible = np.asarray(view.line_of_sight_matrix(origins, targets, candidates))
    mask = (candidates & visible).any(axis=0).reshape(n_rows, n_cols)

    board_w = float(view.config.board_width)
    board_h = float(view.config.board_height)
    rings = _rings_from_mask(mask, spacing, board_w, board_h)
    if smooth <= 0:
        return rings
    return tuple(_chaikin(ring, smooth, board_w, board_h) for ring in rings)


def compute_threat_overlay(
    view: BattleView,
    options: ThreatOptions,
) -> ThreatOverlay:
    """Both overlays for one frame; each half is skipped when switched off."""
    engagement = options.show_engagement
    threat = options.show_threat
    spacing = options.spacing
    smooth = options.smoothing
    return ThreatOverlay(
        player_engagement=(
            compute_engagement_zone(view.player_models) if engagement else ()
        ),
        opponent_engagement=(
            compute_engagement_zone(view.opponent_models) if engagement else ()
        ),
        engagement_radius=engagement_radius(view) if engagement else 0.0,
        player_threat=(
            compute_threat_region(
                view,
                view.player_models,
                view.player_max_ranges,
                spacing=spacing,
                smooth=smooth,
            )
            if threat
            else ()
        ),
        opponent_threat=(
            compute_threat_region(
                view,
                view.opponent_models,
                view.opponent_max_ranges,
                spacing=spacing,
                smooth=smooth,
            )
            if threat and view.opponent_models
            else ()
        ),
    )


def _rings_from_mask(
    mask: np.ndarray, spacing: float, board_w: float, board_h: float
) -> tuple[Ring, ...]:
    """Closed rings tracing the boundary of a ``[row][col]`` True region.

    Every True cell offers its four edges and an edge shared by two True cells
    is dropped; what survives is the region's boundary. Each edge is oriented
    with the True cell on its left, so the survivors chain head-to-tail and
    close by construction — on an integer lattice, so closure is exact and no
    epsilon is involved. Disjoint regions and holes each come back as their own
    ring, which is why an *outline* needs nothing `Poly` cannot express.

    Vertices are **not** collinear-collapsed, and that is the point. Chaikin
    moves a vertex by a quarter of its adjacent segment length, so collapsing a
    twenty-cell run along the board edge into one twenty-inch segment lets
    smoothing haul the boundary five inches inward and swallow the models
    standing on it. Keeping the cell vertices bounds the displacement to an
    eighth of a cell.
    """
    n_rows, n_cols = mask.shape
    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}

    def add(a: tuple[int, int], b: tuple[int, int]) -> None:
        edges.setdefault(a, []).append(b)

    for row in range(n_rows):
        for col in range(n_cols):
            if not mask[row, col]:
                continue
            if row == 0 or not mask[row - 1, col]:
                add((col, row), (col + 1, row))
            if col == n_cols - 1 or not mask[row, col + 1]:
                add((col + 1, row), (col + 1, row + 1))
            if row == n_rows - 1 or not mask[row + 1, col]:
                add((col + 1, row + 1), (col, row + 1))
            if col == 0 or not mask[row, col - 1]:
                add((col, row + 1), (col, row))

    rings: list[Ring] = []
    while edges:
        start = next(iter(edges))
        walk = [start]
        node = start
        while True:
            outgoing = edges.get(node)
            if not outgoing:
                break
            # A saddle vertex (two diagonally-touching True cells) has two ways
            # out; taking them in insertion order resolves it the same way every
            # frame, which is all that is required of the choice.
            nxt = outgoing.pop(0)
            if not outgoing:
                del edges[node]
            if nxt == start:
                break
            walk.append(nxt)
            node = nxt
        if len(walk) >= 3:
            rings.append(
                tuple(
                    (min(x * spacing, board_w), min(y * spacing, board_h))
                    for x, y in walk
                )
            )
    return tuple(rings)


def _chaikin(ring: Ring, iterations: int, board_w: float, board_h: float) -> Ring:
    """Corner-cut a closed ring so a one-inch sample step stops reading as stairs.

    Cosmetic, and the only place the drawing is not literally the engine's
    answer — which is why it is a parameter and why the verification test
    asserts at zero iterations. Measured displacement at two iterations is an
    eighth of a cell, and at one it is exactly zero, because the cut points of a
    rectilinear ring land on the original boundary.

    The result is clamped to the table: a board edge is a real boundary rather
    than a sampling artefact, so a smoothed region must never claim ground off
    the board.
    """
    points = list(ring)
    for _ in range(iterations):
        cut: list[tuple[float, float]] = []
        for i, (ax, ay) in enumerate(points):
            bx, by = points[(i + 1) % len(points)]
            cut.append((ax * 0.75 + bx * 0.25, ay * 0.75 + by * 0.25))
            cut.append((ax * 0.25 + bx * 0.75, ay * 0.25 + by * 0.75))
        points = cut
    return tuple(
        (min(max(x, 0.0), board_w), min(max(y, 0.0), board_h)) for x, y in points
    )


def sight_matrix_from_terrain(
    origins: np.ndarray,
    targets: np.ndarray,
    terrain: Terrain,
    blocking_mask: BlockingMask | None,
    *,
    sample_step: float,
    candidates: np.ndarray | None = None,
) -> np.ndarray:
    """`(P, Q)` sight, traced against terrain the caller already holds.

    The replay adapter's `line_of_sight_matrix`. It goes through the engine's own
    `domain.sight` rather than through anything in `renders/`, so a replayed
    threat region is the same answer as the live one — which is the whole reason
    `build_scene` is shared between the two paths in the first place.

    It lives here because `control.py` is the single v2 -> domain seam; putting
    it in `replay.py` would open a second one.
    """
    return line_of_sight_matrix(
        origins,
        targets,
        terrain,
        blocking_mask,
        sample_step=sample_step,
        candidates=candidates,
    )
