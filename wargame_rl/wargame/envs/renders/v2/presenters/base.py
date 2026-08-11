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
from wargame_rl.wargame.envs.renders.v2.scene import (
    Control,
    HudData,
    Scene,
    build_scene,
)
from wargame_rl.wargame.envs.renders.v2.theme import DEFAULT_THEME, RGB, Theme

GRID_SIZE = 1024  # Longest board side, in pixels, at the fit scale.
# Sizes match the HUD studio mockup (same frame width): small captions, ~20px
# values, a slightly larger hero margin.
_LABEL_SIZE = 13  # Zone captions in the top HUD.
_VALUE_SIZE = 20  # Zone values (VP, held, forces).
_MARGIN_SIZE = 26  # The VP margin — the hero number, a size up.


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
        # The top HUD is two rows tall (a caption row over a value row).
        self._top_h = 2 * theme.north_panel_h
        self._offset_y = self._top_h

    def setup(self, view: BattleView) -> None:
        self._board_w = view.config.board_width
        self._board_h = view.config.board_height
        self._scale = min(GRID_SIZE / self._board_w, GRID_SIZE / self._board_h)
        self._recompute_layout()

    def _recompute_layout(self) -> None:
        self._canvas_w = math.ceil(self._scale * self._board_w)
        self._canvas_h = math.ceil(self._scale * self._board_h)
        north = self._theme.north_panel_h
        self._top_h = 2 * north
        south = self._theme.south_panel_rows * north
        self._window_w = self._canvas_w
        self._window_h = self._canvas_h + self._top_h + south
        self._offset_x = 0
        self._offset_y = self._top_h

    def _scene_for(self, view: BattleView) -> Scene:
        """Build the Scene for a live view (control + debug LOS from the domain)."""
        control = compute_objective_control(view)
        los = probe_debug_los(view) if self._debug_los else None
        return build_scene(
            view,
            control,
            scale=self._scale,
            theme=self._theme,
            debug_los=los,
            show_grid=self._theme.show_grid,
        )

    def _compose_scene(self, scene: Scene) -> Canvas:
        """Rasterise a Scene onto a window-sized frame with the HUD panels.

        Split from `_compose` so the replay presenter can feed a Scene built from
        a recorded snapshot through the identical board + panel pipeline.
        """
        board = self._backend.new_canvas(self._canvas_w, self._canvas_h, scene.board_bg)
        rasterize(self._backend, scene, Camera(self._scale), board)

        frame = self._backend.new_canvas(
            self._window_w, self._window_h, self._theme.palette.window_bg
        )
        self._backend.blit(frame, board, (self._offset_x, self._offset_y))
        self._draw_panels(frame, scene.hud)
        return frame

    def _compose(self, view: BattleView) -> Canvas:
        """Render the board and panels into one window-sized frame."""
        return self._compose_scene(self._scene_for(view))

    # -- panels --------------------------------------------------------------

    def _draw_panels(self, frame: Canvas, hud: HudData) -> None:
        pal = self._theme.palette
        north = self._theme.north_panel_h
        width = self._window_w

        # Top HUD: three zones — objectives | victory points | forces. Two rows,
        # a caption over the value, so the eye lands in the same place each frame.
        self._backend.fill_rect(frame, (0, 0, width, self._top_h), pal.panel_bg)
        self._backend.draw_line(
            frame, (0, self._top_h), (width, self._top_h), pal.panel_line, 1
        )
        label_y = int(self._top_h * 0.34)
        value_y = int(self._top_h * 0.68)
        self._zone_objectives(frame, hud, width // 6, label_y, value_y)
        self._zone_vp(frame, hud, width // 2, label_y, value_y)
        self._zone_forces(frame, hud, 5 * width // 6, label_y, value_y)

        # South: clock/reward row, then the hotkey hints.
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
        fields: tuple[tuple[str, int], ...]
        if hud.epoch is not None:
            fields = (
                (f"Epoch: {hud.epoch}", width // 8),
                (turn, 3 * width // 8),
                (steps, 5 * width // 8),
                (reward_text, 7 * width // 8),
            )
        else:
            # Inset from width/6 so the wider monospace text clears the edges.
            fields = (
                (turn, width // 5),
                (steps, width // 2),
                (reward_text, 4 * width // 5),
            )
        for text, cx in fields:
            self._text(frame, text, (cx, row1_y), _VALUE_SIZE, pal.text)

        row2_y = panel_y + north + north // 2
        hotkeys = (
            "PAUSED - Space: Resume | Esc: Quit | L: LOS debug"
            if self._is_paused()
            else "Space: Pause | Esc: Quit | L: LOS debug"
        )
        self._text(frame, hotkeys, (width // 2, row2_y), _LABEL_SIZE + 3, self._dim())

    # -- top HUD zones -------------------------------------------------------

    def _dim(self) -> RGB:
        """A caption colour halfway between the panel text and its background."""
        pal = self._theme.palette
        return (
            (pal.text[0] + pal.panel_bg[0]) // 2,
            (pal.text[1] + pal.panel_bg[1]) // 2,
            (pal.text[2] + pal.panel_bg[2]) // 2,
        )

    def _text(
        self,
        frame: Canvas,
        text: str,
        anchor: tuple[float, float],
        size: int,
        color: RGB,
        align: str = "center",
        bold: bool = False,
    ) -> None:
        """Draw HUD text in the monospace (tabular) face."""
        self._backend.draw_text(
            frame, text, anchor, size, color, align, mono=True, bold=bold
        )

    def _tsize(self, text: str, size: int, bold: bool = False) -> tuple[int, int]:
        """Measure HUD text in the monospace face (for fixed-field layout)."""
        return self._backend.text_size(text, size, mono=True, bold=bold)

    def _draw_bar(
        self, frame: Canvas, x: int, y: int, w: int, h: int, frac: float, color: RGB
    ) -> None:
        pal = self._theme.palette
        top = int(y - h / 2)
        self._backend.fill_rect(frame, (x, top, w, h), pal.panel_line)
        fill_w = max(0, min(w, int(w * frac)))
        if fill_w > 0:
            self._backend.fill_rect(frame, (x, top, fill_w, h), color)

    def _zone_objectives(
        self, frame: Canvas, hud: HudData, cx: int, label_y: int, value_y: int
    ) -> None:
        pal = self._theme.palette
        self._text(frame, "OBJECTIVES", (cx, label_y), _LABEL_SIZE, self._dim())
        pip_gap = 20
        pips_w = len(hud.objective_controls) * pip_gap
        # "Held P-O" with the two counts coloured by side.
        prefix_w = self._tsize("Held ", _VALUE_SIZE, bold=True)[0]
        digit_w = self._tsize("0", _VALUE_SIZE, bold=True)[0]
        dash_w = self._tsize("-", _VALUE_SIZE, bold=True)[0]
        held_w = prefix_w + digit_w + dash_w + digit_w
        gap = 16
        left = cx - (pips_w + gap + held_w) // 2
        x = left + pip_gap // 2
        for control in hud.objective_controls:
            color = (
                pal.player_control
                if control is Control.PLAYER
                else pal.opponent_control
                if control is Control.OPPONENT
                else pal.objective_rim
            )
            self._backend.draw_disc(
                frame, (x, value_y), 6, (*color, 255), pal.panel_line, 1
            )
            x += pip_gap
        hx = left + pips_w + gap
        self._text(
            frame, "Held ", (hx, value_y), _VALUE_SIZE, pal.text, "midleft", True
        )
        self._text(
            frame,
            str(hud.held_player),
            (hx + prefix_w, value_y),
            _VALUE_SIZE,
            pal.hud_player,
            "midleft",
            True,
        )
        self._text(
            frame,
            "-",
            (hx + prefix_w + digit_w, value_y),
            _VALUE_SIZE,
            pal.text,
            "midleft",
            True,
        )
        self._text(
            frame,
            str(hud.held_opponent),
            (hx + prefix_w + digit_w + dash_w, value_y),
            _VALUE_SIZE,
            pal.hud_opponent,
            "midleft",
            True,
        )

    def _zone_vp(
        self, frame: Canvas, hud: HudData, cx: int, label_y: int, value_y: int
    ) -> None:
        pal = self._theme.palette
        self._text(frame, "VICTORY POINTS", (cx, label_y), _LABEL_SIZE, self._dim())
        margin = hud.player_vp - hud.opponent_vp
        margin_text = f"+{margin}" if margin > 0 else str(margin)
        margin_color = (
            pal.hud_player
            if margin > 0
            else pal.hud_opponent
            if margin < 0
            else pal.text
        )
        # Fixed fields so a digit or sign never shifts the readout; the margin is
        # the hero — a size up and bold.
        vp_field = self._tsize("00", _VALUE_SIZE, bold=True)[0]
        margin_field = self._tsize("-00", _MARGIN_SIZE, bold=True)[0]
        gap = self._tsize("  ", _VALUE_SIZE)[0]
        left = cx - (vp_field + gap + margin_field + gap + vp_field) // 2
        self._text(
            frame,
            str(hud.player_vp),
            (left + vp_field, value_y),
            _VALUE_SIZE,
            pal.hud_player,
            "midright",
            True,
        )
        margin_cx = left + vp_field + gap + margin_field // 2
        self._text(
            frame,
            margin_text,
            (margin_cx, value_y),
            _MARGIN_SIZE,
            margin_color,
            "center",
            True,
        )
        opp_left = left + vp_field + gap + margin_field + gap
        self._text(
            frame,
            str(hud.opponent_vp),
            (opp_left, value_y),
            _VALUE_SIZE,
            pal.hud_opponent,
            "midleft",
            True,
        )

    def _zone_forces(
        self, frame: Canvas, hud: HudData, cx: int, label_y: int, value_y: int
    ) -> None:
        pal = self._theme.palette
        self._text(frame, "FORCES", (cx, label_y), _LABEL_SIZE, self._dim())
        player = pal.hud_player
        opponent = pal.hud_opponent
        player_text = f"{hud.player_alive}/{hud.player_total}"
        opp_text = f"{hud.opponent_alive}/{hud.opponent_total}"
        text_field = self._tsize("00/00", _VALUE_SIZE)[0]
        bar_w, bar_h, gap, mid = 58, 8, 9, 16
        left = cx - (2 * text_field + 2 * gap + 2 * bar_w + mid) // 2
        self._text(
            frame,
            player_text,
            (left + text_field, value_y),
            _VALUE_SIZE,
            player,
            "midright",
        )
        bx = left + text_field + gap
        self._draw_bar(
            frame,
            bx,
            value_y,
            bar_w,
            bar_h,
            hud.player_alive / max(1, hud.player_total),
            player,
        )
        bx2 = bx + bar_w + mid
        self._draw_bar(
            frame,
            bx2,
            value_y,
            bar_w,
            bar_h,
            hud.opponent_alive / max(1, hud.opponent_total),
            opponent,
        )
        self._text(
            frame,
            opp_text,
            (bx2 + bar_w + gap, value_y),
            _VALUE_SIZE,
            opponent,
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
