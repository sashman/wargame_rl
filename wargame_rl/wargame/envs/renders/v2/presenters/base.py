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
    shot_fade_for_age,
)
from wargame_rl.wargame.envs.renders.v2.theme import DEFAULT_THEME, RGB, Theme

GRID_SIZE = 1024  # Longest board side, in pixels, at the fit scale.
# Sizes match the HUD studio mockup (same frame width): small captions, ~20px
# values, a slightly larger hero margin.
_LABEL_SIZE = 13  # Zone captions in the top HUD.
_VALUE_SIZE = 20  # Zone values (VP, held, forces).
_MARGIN_SIZE = 26  # The VP margin — the hero number, a size up.
_CHIP_SIZE = 12  # Phase chain + reward summary, the smallest HUD text.
_SOUTH_MARGIN = 16  # Inset of the south panel's outer columns.
# The round track holds this width whatever `number_of_battle_rounds` is: one
# segment per round while a segment stays wider than the gaps between them,
# a continuous fill past that. Nothing in the panel scales with the config.
_TRACK_W = 170
_TRACK_H = 7
_TRACK_GAP = 3
_MIN_SEGMENT = 3
_COMP_BAR_W = 260  # Reward-composition bar: one segment per calculator.
_COMP_BAR_H = 9
_MIN_COMP_SEGMENT = 2  # A tiny non-zero term still has to be visible.
_KEY_MAP_KEY = "Tab"  # Opens the key map; the only key the panel itself names.
_KEY_ROW_H = 26
_KEY_SIZE = 15
_KEY_TITLE_SIZE = 17


def _blend(a: RGB, b: RGB) -> RGB:
    """The midpoint of two colours."""
    return ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2, (a[2] + b[2]) // 2)


class BasePresenter(Renderer):
    """Scene→frame pipeline shared by the interactive and recording presenters."""

    def __init__(self, backend: RenderBackend, theme: Theme = DEFAULT_THEME) -> None:
        self._backend = backend
        self._theme = theme
        self.epoch: int | None = None
        # Free-text provenance for the context slot (the training run's name, a
        # config stem, a seed) — set by whoever drives the renderer.
        self.run_label: str | None = None
        self._debug_los = False
        # Volleys fade over a few frames, so the presenter has to know how long
        # ago the current results resolved. The env keeps the last results until
        # the next shooting phase, so age is counted here rather than read off it.
        self._shot_age = 0
        self._shot_signature: tuple[object, ...] = ()
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

    def _reserved_width(self) -> int:
        """Window width claimed by something other than the board.

        Zero for every presenter that draws only the board and the two HUD rows.
        The debug presenter reserves a side panel here, which is what keeps the
        board from being drawn underneath it.
        """
        return 0

    def _recompute_layout(self) -> None:
        self._canvas_w = math.ceil(self._scale * self._board_w)
        self._canvas_h = math.ceil(self._scale * self._board_h)
        north = self._theme.north_panel_h
        self._top_h = 2 * north
        south = self._theme.south_panel_rows * north
        self._window_w = self._canvas_w + self._reserved_width()
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
            shot_fade=shot_fade_for_age(self._age_of_volley(view)),
        )

    def _age_of_volley(self, view: BattleView) -> int:
        """Frames since the currently-reported shooting results first appeared.

        Identity is not enough — the env hands out a fresh list each call — so
        the results are compared by content. Two identical volleys in a row would
        read as one, which costs a re-flash and nothing else.
        """
        signature = tuple(
            (r.attacker_idx, r.target_idx, r.result.damage_dealt, r.killed)
            for r in (
                *view.last_player_shooting_results,
                *view.last_opponent_shooting_results,
            )
        )
        if signature != self._shot_signature:
            self._shot_signature = signature
            self._shot_age = 0
        else:
            self._shot_age += 1
        return self._shot_age

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

        # South HUD: the same three columns as the top, but for *time*, *reward*
        # and *context* — clock left, reward centre, run/pause state right, with
        # the phase chain and the reward's components on the second row.
        south_h = self._theme.south_panel_rows * north
        panel_y = self._window_h - south_h
        self._backend.fill_rect(frame, (0, panel_y, width, south_h), pal.panel_bg)
        self._backend.draw_line(
            frame, (0, panel_y), (width, panel_y), pal.panel_line, 1
        )

        row1_y = panel_y + north // 2
        row2_y = panel_y + north + north // 2
        self._south_clock(frame, hud, _SOUTH_MARGIN, row1_y)
        self._south_phases(frame, hud, _SOUTH_MARGIN, row2_y)
        self._south_reward(frame, hud, width // 2, row1_y, row2_y)
        self._south_context(frame, width - _SOUTH_MARGIN, row1_y)
        hint = self._hotkey_hint()
        if hint:
            self._text(
                frame,
                hint,
                (width - _SOUTH_MARGIN, row2_y),
                _CHIP_SIZE,
                self._dim(),
                "midright",
            )

    # -- key map overlay -----------------------------------------------------

    def _draw_key_map(self, frame: Canvas) -> None:
        """The full key list, centred over the frame.

        Lives here rather than in the panel because the list grows with every
        key added, while the panel's width does not — the same reason the reward
        ledger is summarised down there instead of enumerated.
        """
        keys = self.key_map()
        if not keys:
            return
        pal = self._theme.palette
        key_col = max(
            self._tsize(f"[{key}]", _KEY_SIZE, bold=True)[0] for key, _ in keys
        )
        action_col = max(self._tsize(action, _KEY_SIZE)[0] for _, action in keys)
        pad = 22
        gap = 16
        width = key_col + gap + action_col + 2 * pad
        height = (len(keys) + 2) * _KEY_ROW_H + pad
        x = (self._window_w - width) // 2
        y = (self._window_h - height) // 2

        # Dim the frame behind it so the map reads as a layer, not as more HUD.
        self._backend.draw_polygon(
            frame,
            [
                (0, 0),
                (self._window_w, 0),
                (self._window_w, self._window_h),
                (0, self._window_h),
            ],
            (*pal.window_bg, 190),
            None,
            0,
        )
        self._backend.fill_rect(frame, (x - 1, y - 1, width + 2, height + 2), pal.text)
        self._backend.fill_rect(frame, (x, y, width, height), pal.panel_bg)

        row_y = y + pad + _KEY_ROW_H // 2
        self._text(
            frame, "KEYS", (x + pad, row_y), _KEY_TITLE_SIZE, pal.text, "midleft", True
        )
        self._text(
            frame,
            f"[{_KEY_MAP_KEY}] closes",
            (x + width - pad, row_y),
            _CHIP_SIZE,
            self._dim(),
            "midright",
        )
        row_y += _KEY_ROW_H
        self._backend.draw_line(
            frame, (x + pad, row_y), (x + width - pad, row_y), pal.panel_line, 1
        )
        for key, action in keys:
            row_y += _KEY_ROW_H
            self._text(
                frame,
                f"[{key}]",
                (x + pad, row_y),
                _KEY_SIZE,
                pal.hud_player,
                "midleft",
                True,
            )
            self._text(
                frame,
                action,
                (x + pad + key_col + gap, row_y),
                _KEY_SIZE,
                pal.text,
                "midleft",
            )

    # -- south HUD zones -----------------------------------------------------

    def _draw_clock_icon(self, frame: Canvas, cx: int, cy: int, color: RGB) -> None:
        """A clock face drawn from primitives.

        Not a glyph: the resolved monospace face is whatever the system has, and
        a missing character would render as a box in the one place the HUD says
        what time it is.
        """
        radius = 7
        self._backend.draw_disc(frame, (cx, cy), radius, None, color, 1)
        self._backend.draw_line(frame, (cx, cy), (cx, cy - radius + 2), color, 1)
        self._backend.draw_line(frame, (cx, cy), (cx + radius - 3, cy + 2), color, 1)

    def _south_clock(self, frame: Canvas, hud: HudData, x: int, y: int) -> None:
        """Clock icon, `R 3/5`, and a track of the rounds played so far.

        The round is right-aligned in a field as wide as the round *count*, so
        reaching round 10 of 20 neither widens the readout nor shoves the track
        along — the same fixed-slot rule the top HUD's numbers follow.
        """
        pal = self._theme.palette
        self._draw_clock_icon(frame, x + 7, y, pal.text)
        text_x = x + 22
        total = f"/{hud.n_rounds}"
        prefix_w = self._tsize("R ", _VALUE_SIZE, bold=True)[0]
        count_w = self._tsize("0" * len(str(hud.n_rounds)), _VALUE_SIZE, bold=True)[0]
        self._text(frame, "R", (text_x, y), _VALUE_SIZE, pal.text, "midleft", True)
        self._text(
            frame,
            str(hud.round),
            (text_x + prefix_w + count_w, y),
            _VALUE_SIZE,
            pal.text,
            "midright",
            True,
        )
        self._text(
            frame,
            total,
            (text_x + prefix_w + count_w, y),
            _VALUE_SIZE,
            pal.text,
            "midleft",
            True,
        )
        left = (
            text_x
            + prefix_w
            + count_w
            + self._tsize(total, _VALUE_SIZE, bold=True)[0]
            + 12
        )
        self._draw_round_track(frame, left, y, hud.round, hud.n_rounds)
        after_turn = self._draw_turn_order(frame, hud, left + _TRACK_W + 14, y)
        self._draw_undo_depth(frame, hud, after_turn + 16, y)

    def _draw_undo_depth(self, frame: Canvas, hud: HudData, x: int, y: int) -> None:
        """How many steps are available to rewind, labelled with the key itself.

        Dim at zero, which is the whole point: pressing the key then is a no-op,
        and the readout says so *before* it is pressed rather than logging it
        somewhere the user is not looking. Right-aligned in a three-digit slot so
        the field does not jitter as the history grows.
        """
        if hud.undo_depth is None:
            return
        pal = self._theme.palette
        self._text(frame, "[,]", (x, y), _CHIP_SIZE, self._dim(), "midleft")
        key_w = self._tsize("[,] ", _CHIP_SIZE)[0]
        slot_w = self._tsize("000", _CHIP_SIZE)[0]
        self._text(
            frame,
            str(hud.undo_depth),
            (x + key_w + slot_w, y),
            _CHIP_SIZE,
            pal.text if hud.undo_depth else self._dim(),
            "midright",
        )

    def _draw_turn_order(self, frame: Canvas, hud: HudData, x: int, y: int) -> int:
        """Who takes the first turn of each round, in a fixed two-slot field.

        Deliberately *order* rather than "whose turn is it now": the opponent's
        whole turn runs inside the player's `step()`, so every frame ever drawn
        is one where the player is next and a live indicator would be a constant.
        Order is the part that varies — under `turn_order: random` the player's
        side is re-rolled each reset — and it answers the question that matters
        while paused: whether the positions on screen already include the
        opponent's reply to this round, or come before it.

        Returns the x the field ends at, so what follows can sit beside it.
        """
        if hud.player_acts_first is None:
            return x
        pal = self._theme.palette
        first, second = (
            (("YOU", pal.hud_player), ("OPP", pal.hud_opponent))
            if hud.player_acts_first
            else (("OPP", pal.hud_opponent), ("YOU", pal.hud_player))
        )
        self._text(frame, first[0], (x, y), _CHIP_SIZE, first[1], "midleft", True)
        arrow_x = x + self._tsize("YOU ", _CHIP_SIZE, bold=True)[0]
        self._text(frame, ">", (arrow_x, y), _CHIP_SIZE, self._dim(), "midleft")
        second_x = arrow_x + self._tsize("> ", _CHIP_SIZE)[0]
        self._text(frame, second[0], (second_x, y), _CHIP_SIZE, second[1], "midleft")
        return second_x + self._tsize(second[0], _CHIP_SIZE)[0]

    def _draw_round_track(
        self, frame: Canvas, x: int, y: int, played: int, total: int
    ) -> None:
        """Rounds elapsed, in a fixed width however many rounds the config has.

        Segments while they stay wider than the gaps between them; beyond that a
        continuous fill, which says the same thing without turning into a blur.
        """
        pal = self._theme.palette
        top = int(y - _TRACK_H / 2)
        rounds = max(1, total)
        segment_w = (_TRACK_W - _TRACK_GAP * (rounds - 1)) / rounds
        if segment_w < _MIN_SEGMENT:
            self._backend.fill_rect(frame, (x, top, _TRACK_W, _TRACK_H), pal.panel_line)
            filled = int(_TRACK_W * min(1.0, played / rounds))
            if filled > 0:
                self._backend.fill_rect(frame, (x, top, filled, _TRACK_H), pal.text)
            return
        for index in range(rounds):
            color = pal.text if index < played else pal.panel_line
            left = x + int(index * (segment_w + _TRACK_GAP))
            self._backend.fill_rect(
                frame, (left, top, max(1, int(segment_w)), _TRACK_H), color
            )

    def _south_reward(
        self, frame: Canvas, hud: HudData, cx: int, y: int, bar_y: int
    ) -> None:
        """The step's reward as the hero, over its composition and a paid/charged count."""
        pal = self._theme.palette
        step_text = f"{hud.reward:+.3f}" if hud.reward is not None else "—"
        color = (
            pal.text
            if not hud.reward
            else pal.hud_player
            if hud.reward > 0
            else pal.hud_opponent
        )
        total_text = (
            f"total {hud.episode_reward:+.1f}" if hud.episode_reward is not None else ""
        )
        # Fixed fields so a sign or a digit never shifts the readout.
        step_field = self._tsize("-00.000", _MARGIN_SIZE, bold=True)[0]
        total_field = self._tsize("total -000.0", _CHIP_SIZE)[0]
        gap = 14
        left = cx - (step_field + gap + total_field) // 2
        self._text(
            frame,
            step_text,
            (left + step_field, y),
            _MARGIN_SIZE,
            color,
            "midright",
            True,
        )
        if total_text:
            self._text(
                frame,
                total_text,
                (left + step_field + gap, y),
                _CHIP_SIZE,
                self._dim(),
                "midleft",
            )
        self._draw_composition_bar(frame, hud, cx, bar_y)
        self._draw_income_summary(frame, hud, cx, bar_y)

    def _draw_composition_bar(
        self, frame: Canvas, hud: HudData, cx: int, y: int
    ) -> None:
        """One segment per reward component: width is magnitude, colour is sign.

        This is what makes the panel independent of the reward config — eleven
        calculators is eleven thinner segments, not a row that overflows.
        """
        pal = self._theme.palette
        parts = [(name, value) for name, value in hud.reward_breakdown if value]
        left = cx - _COMP_BAR_W // 2
        top = int(y - _COMP_BAR_H / 2) - 9
        self._backend.fill_rect(
            frame, (left, top, _COMP_BAR_W, _COMP_BAR_H), pal.panel_line
        )
        if not parts:
            return
        magnitude = sum(abs(value) for _, value in parts)
        gap = 2
        available = _COMP_BAR_W - gap * (len(parts) - 1)
        x = left
        for _, value in parts:
            # A floor, so a small non-zero term reads as present rather than absent.
            seg_w = max(_MIN_COMP_SEGMENT, int(available * abs(value) / magnitude))
            color = pal.hud_player if value > 0 else pal.hud_opponent
            self._backend.fill_rect(frame, (x, top, seg_w, _COMP_BAR_H), color)
            x += seg_w + gap

    def _draw_income_summary(
        self, frame: Canvas, hud: HudData, cx: int, y: int
    ) -> None:
        """`+0.43 from 4   -0.05 from 2` — the ledger's two totals, never more."""
        pal = self._theme.palette
        paid = [value for _, value in hud.reward_breakdown if value > 0]
        charged = [value for _, value in hud.reward_breakdown if value < 0]
        texts = [
            (f"{sum(values):+.2f} from {len(values)}", color)
            for values, color in ((paid, pal.hud_player), (charged, pal.hud_opponent))
            if values
        ]
        if not texts:
            return
        gap = self._tsize("   ", _CHIP_SIZE)[0]
        total_w = sum(self._tsize(t, _CHIP_SIZE)[0] for t, _ in texts)
        total_w += gap * (len(texts) - 1)
        x = cx - total_w // 2
        for text, color in texts:
            self._text(frame, text, (x, y + 8), _CHIP_SIZE, color, "midleft")
            x += self._tsize(text, _CHIP_SIZE)[0] + gap

    def _south_context(self, frame: Canvas, x: int, y: int) -> None:
        """Who/what produced this frame: the run label and epoch, or PAUSED.

        Interactively the pause state is what matters; in a recording nobody can
        press a key, so the slot stamps the video with its own provenance.
        """
        pal = self._theme.palette
        if self._is_paused():
            self._text(frame, "PAUSED", (x, y), _VALUE_SIZE, pal.text, "midright", True)
            return
        parts = [part for part in (self.run_label, self._epoch_text()) if part]
        if not parts:
            return
        self._text(frame, "  ".join(parts), (x, y), _CHIP_SIZE, self._dim(), "midright")

    def _epoch_text(self) -> str:
        return f"epoch {self.epoch}" if self.epoch is not None else ""

    def _south_phases(self, frame: Canvas, hud: HudData, x: int, y: int) -> None:
        """The round's phase chain; skipped phases are faint, the current one lit."""
        pal = self._theme.palette
        dim = self._dim()
        faint = _blend(dim, pal.panel_bg)
        gap = self._tsize(" ", _CHIP_SIZE)[0] * 2
        for chip in hud.phase_chips:
            color = pal.text if chip.is_current else faint if chip.is_skipped else dim
            self._text(
                frame, chip.label, (x, y), _CHIP_SIZE, color, "midleft", chip.is_current
            )
            x += self._tsize(chip.label, _CHIP_SIZE, bold=True)[0] + gap

    # -- top HUD zones -------------------------------------------------------

    def _dim(self) -> RGB:
        """A caption colour halfway between the panel text and its background."""
        pal = self._theme.palette
        return _blend(pal.text, pal.panel_bg)

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

    def key_map(self) -> tuple[tuple[str, str], ...]:
        """This presenter's keys as `(key, what it does)`, for the Tab overlay.

        Empty for a presenter nobody can type at, which is also what hides the
        hint from a recording.
        """
        return ()

    def _hotkey_hint(self) -> str | None:
        """The one hint the panel carries; the rest live behind it."""
        return f"[{_KEY_MAP_KEY}] keys" if self.key_map() else None

    def _present(self, frame: Canvas) -> None:
        raise NotImplementedError

    def render(self, view: BattleView) -> None:
        self._present(self._compose(view))

    def close(self) -> None:
        pass
