"""Shared presenter: build a Scene, rasterise the board, compose the HUD panels.

Subclasses provide only the window/output half (`_present`) and, for interactive
use, the event loop. The board is drawn to its own canvas at the fit scale and
composited into a window-sized frame under the north panel; the panels are drawn
with the same backend primitives so a recording includes them.
"""

from __future__ import annotations

import math

from wargame_rl.wargame.envs.domain.battle_view import BattleView
from wargame_rl.wargame.envs.renders.renderer import Renderer
from wargame_rl.wargame.envs.renders.v2.backend import Canvas, RenderBackend, rasterize
from wargame_rl.wargame.envs.renders.v2.camera import Camera
from wargame_rl.wargame.envs.renders.v2.control import (
    compute_objective_control,
    probe_debug_los,
)
from wargame_rl.wargame.envs.renders.v2.scene import HudData, build_scene
from wargame_rl.wargame.envs.renders.v2.theme import DEFAULT_THEME, Theme

GRID_SIZE = 1024  # Longest board side, in pixels, at the fit scale.


class BasePresenter(Renderer):
    """Scene→frame pipeline shared by the interactive and recording presenters."""

    def __init__(self, backend: RenderBackend, theme: Theme = DEFAULT_THEME) -> None:
        self._backend = backend
        self._theme = theme
        self.epoch: int | None = None
        self._debug_los = False
        # Board fit + window layout, set in `setup`.
        self._scale = 1.0
        self._board_w = 1
        self._board_h = 1
        self._canvas_w = 1
        self._canvas_h = 1
        self._window_w = 1
        self._window_h = 1
        self._offset_x = 0
        self._offset_y = theme.north_panel_h

    def setup(self, view: BattleView) -> None:
        self._board_w = view.config.board_width
        self._board_h = view.config.board_height
        self._scale = min(GRID_SIZE / self._board_w, GRID_SIZE / self._board_h)
        self._recompute_layout()

    def _recompute_layout(self) -> None:
        self._canvas_w = math.ceil(self._scale * self._board_w)
        self._canvas_h = math.ceil(self._scale * self._board_h)
        north = self._theme.north_panel_h
        south = self._theme.south_panel_rows * north
        self._window_w = self._canvas_w
        self._window_h = self._canvas_h + north + south
        self._offset_x = 0
        self._offset_y = north

    def _compose(self, view: BattleView) -> Canvas:
        """Render the board and panels into one window-sized frame."""
        control = compute_objective_control(view)
        los = probe_debug_los(view) if self._debug_los else None
        scene = build_scene(
            view,
            control,
            scale=self._scale,
            theme=self._theme,
            debug_los=los,
            show_grid=self._theme.show_grid,
        )

        board = self._backend.new_canvas(self._canvas_w, self._canvas_h, scene.board_bg)
        rasterize(self._backend, scene, Camera(self._scale), board)

        frame = self._backend.new_canvas(
            self._window_w, self._window_h, self._theme.palette.window_bg
        )
        self._backend.blit(frame, board, (self._offset_x, self._offset_y))
        self._draw_panels(frame, scene.hud)
        return frame

    # -- panels --------------------------------------------------------------

    def _draw_panels(self, frame: Canvas, hud: HudData) -> None:
        pal = self._theme.palette
        north = self._theme.north_panel_h
        width = self._window_w

        # North: hotkey bar.
        self._backend.fill_rect(frame, (0, 0, width, north), pal.panel_bg)
        self._backend.draw_line(frame, (0, north), (width, north), pal.panel_line, 1)
        hotkeys = (
            "PAUSED - Space: Resume | Esc: Quit | L: LOS debug"
            if self._is_paused()
            else "Space: Pause | Esc: Quit | L: LOS debug"
        )
        self._backend.draw_text(frame, hotkeys, (width // 2, north // 2), 24, pal.text)

        # South: two info rows.
        south_h = self._theme.south_panel_rows * north
        panel_y = self._window_h - south_h
        self._backend.fill_rect(frame, (0, panel_y, width, south_h), pal.panel_bg)
        self._backend.draw_line(
            frame, (0, panel_y), (width, panel_y), pal.panel_line, 1
        )

        row1_y = panel_y + north // 2
        reward = f"{hud.reward:.3f}" if hud.reward is not None else "—"
        turn = f"Round: {hud.round} / {hud.n_rounds}  |  {hud.phase}"
        steps = f"Step: {hud.step}"
        reward_text = f"Reward: {reward}"
        if hud.epoch is not None:
            for text, cx in (
                (f"Epoch: {hud.epoch}", width // 8),
                (turn, 3 * width // 8),
                (steps, 5 * width // 8),
                (reward_text, 7 * width // 8),
            ):
                self._backend.draw_text(frame, text, (cx, row1_y), 24, pal.text)
        else:
            for text, cx in (
                (turn, width // 6),
                (steps, width // 2),
                (reward_text, 5 * width // 6),
            ):
                self._backend.draw_text(frame, text, (cx, row1_y), 24, pal.text)

        row2_y = panel_y + north + north // 2
        self._draw_vp(
            frame, "Player VP:", hud.player_vp, hud.player_vp_delta, width // 4, row2_y
        )
        self._draw_vp(
            frame,
            "Opponent VP:",
            hud.opponent_vp,
            hud.opponent_vp_delta,
            3 * width // 4,
            row2_y,
        )

    def _draw_vp(
        self,
        frame: Canvas,
        label: str,
        value: int,
        delta: int,
        center_x: int,
        center_y: int,
    ) -> None:
        """Fixed-field VP readout so nothing shifts when the delta appears."""
        pal = self._theme.palette
        gap = self._backend.text_size(" ", 24)[0]
        value_field = self._backend.text_size("000", 24)[0]
        delta_field = self._backend.text_size("(+00)", 24)[0]
        label_w = self._backend.text_size(label, 24)[0]

        total = label_w + gap + value_field + gap + delta_field
        left = center_x - total // 2
        self._backend.draw_text(frame, label, (left, center_y), 24, pal.text, "midleft")
        value_right = left + label_w + gap + value_field
        self._backend.draw_text(
            frame, str(value), (value_right, center_y), 24, pal.text, "midright"
        )
        if delta > 0:
            self._backend.draw_text(
                frame,
                f"(+{delta})",
                (value_right + gap, center_y),
                24,
                pal.text,
                "midleft",
            )

    # -- hooks ---------------------------------------------------------------

    def _is_paused(self) -> bool:
        return False

    def _present(self, frame: Canvas) -> None:
        raise NotImplementedError

    def render(self, view: BattleView) -> None:
        self._present(self._compose(view))

    def close(self) -> None:
        pass
