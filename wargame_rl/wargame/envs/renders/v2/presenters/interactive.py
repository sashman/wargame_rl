"""Interactive presenter: a resizable pygame window with the legacy controls.

Space pauses, Esc quits (raising `QuitRequested`, as the legacy renderer did),
L toggles the debug sight line, a left click pins a model's tooltip, Tab shows
the key map, and resizing refits the board. The keys themselves are declared once
in `key_map`, which is what the overlay and the panel's hint both read. Pan/zoom
is deliberately absent in Phase 1 — the `Camera` is where it will slot in later.
"""

from __future__ import annotations

import pygame

from wargame_rl.wargame.envs.domain.battle_view import BattleView
from wargame_rl.wargame.envs.renders.human import QuitRequested
from wargame_rl.wargame.envs.renders.v2.backend import Canvas, RenderBackend
from wargame_rl.wargame.envs.renders.v2.presenters.base import BasePresenter
from wargame_rl.wargame.envs.renders.v2.theme import DEFAULT_THEME, Theme


class InteractiveRenderer(BasePresenter):
    """Drives a live pygame window."""

    def __init__(self, backend: RenderBackend, theme: Theme = DEFAULT_THEME) -> None:
        super().__init__(backend, theme)
        self._window: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._paused = False
        self._should_quit = False
        self._pinned: int | None = None
        self._fps = 5
        self._show_keys = False

    def setup(self, view: BattleView) -> None:
        super().setup(view)
        self._fps = int(view.metadata.get("render_fps", 5))
        if self._window is None:
            pygame.init()
            pygame.display.init()
            self._window = pygame.display.set_mode(
                (self._window_w, self._window_h), pygame.RESIZABLE
            )
            pygame.display.set_caption("Wargame (v2)")
        else:
            self._window = pygame.display.set_mode(
                (self._window_w, self._window_h), pygame.RESIZABLE
            )
        if self._clock is None:
            self._clock = pygame.time.Clock()

    def _is_paused(self) -> bool:
        return self._paused

    def key_map(self) -> tuple[tuple[str, str], ...]:
        return (
            ("Space", "pause / resume"),
            ("L", "line-of-sight debug ray"),
            ("Click", "pin a model's tooltip"),
            ("Esc", "quit"),
        )

    def render(self, view: BattleView) -> None:
        self._process_events(view)
        if self._should_quit:
            raise QuitRequested()
        self._present(self._compose_with_tooltip(view))
        while self._paused:
            self._process_events(view)
            if self._should_quit:
                raise QuitRequested()
            self._present(self._compose_with_tooltip(view))

    def _compose_with_tooltip(self, view: BattleView) -> Canvas:
        frame = self._compose(view)
        index = self._pinned if self._pinned is not None else self._hovered(view)
        if index is not None:
            self._draw_tooltip(frame, view, index)
        if self._show_keys:
            self._draw_key_map(frame)
        return frame

    def _present(self, frame: Canvas) -> None:
        if self._window is None or self._clock is None:
            return
        # Go through the backend's RGB array so the window works for any backend,
        # not just pygame — a Pillow canvas is not a blittable Surface. `frombuffer`
        # reads the row-major (H, W, 3) buffer directly, so no transpose is needed.
        rgb = self._backend.to_rgb_array(frame)
        surface = pygame.image.frombuffer(
            rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB"
        )
        self._window.blit(surface, (0, 0))
        pygame.event.pump()
        pygame.display.update()
        self._clock.tick(self._fps)

    # -- events --------------------------------------------------------------

    def _process_events(self, view: BattleView) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._should_quit = True
            elif event.type == pygame.VIDEORESIZE:
                self._fit_to_window(view, event.w, event.h)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self._paused = not self._paused
                elif event.key == pygame.K_ESCAPE:
                    self._should_quit = True
                elif event.key == pygame.K_l:
                    self._debug_los = not self._debug_los
                elif event.key == pygame.K_TAB:
                    self._show_keys = not self._show_keys
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._pinned = self._model_index_at(view, event.pos[0], event.pos[1])

    def _fit_to_window(self, view: BattleView, w: int, h: int) -> None:
        top = 2 * self._theme.north_panel_h  # two-row top HUD
        south = self._theme.south_panel_rows * self._theme.north_panel_h
        new_w = max(1, w)
        new_h = max(top + south + 1, h)
        if self._window is not None and (new_w, new_h) == self._window.get_size():
            return
        self._window = pygame.display.set_mode((new_w, new_h), pygame.RESIZABLE)
        available_h = max(1, new_h - top - south)
        self._scale = min(new_w / self._board_w, available_h / self._board_h)
        import math

        self._canvas_w = math.ceil(self._scale * self._board_w)
        self._canvas_h = math.ceil(self._scale * self._board_h)
        self._top_h = top
        self._window_w = new_w
        self._window_h = new_h
        self._offset_x = (new_w - self._canvas_w) // 2
        self._offset_y = top + (available_h - self._canvas_h) // 2

    # -- tooltip -------------------------------------------------------------

    def _model_index_at(self, view: BattleView, mx: int, my: int) -> int | None:
        if not (
            self._offset_x <= mx < self._offset_x + self._canvas_w
            and self._offset_y <= my < self._offset_y + self._canvas_h
        ):
            return None
        canvas_x = float(mx - self._offset_x)
        canvas_y = float(my - self._offset_y)
        hit_radius = max(self._scale / 2, 12.0)
        for i, model in enumerate(view.player_models):
            cx = model.location[0] * self._scale
            cy = model.location[1] * self._scale
            if (canvas_x - cx) ** 2 + (canvas_y - cy) ** 2 <= hit_radius**2:
                return i
        return None

    def _hovered(self, view: BattleView) -> int | None:
        mx, my = pygame.mouse.get_pos()
        return self._model_index_at(view, mx, my)

    def _draw_tooltip(self, frame: Canvas, view: BattleView, index: int) -> None:
        model = view.player_models[index]
        latest = (
            model.model_rewards_history[-1] if model.model_rewards_history else None
        )
        lines = [
            f"Location: ({model.location[0]}, {model.location[1]})",
            f"Group ID: {model.group_id}",
        ]
        if latest is not None:
            lines += [
                f"Closest objective reward: {latest.closest_objective_reward:.3f}",
                f"Group distance penalty: {latest.group_distance_violation_penalty:.3f}",
                f"Total reward: {latest.total_reward:.3f}",
            ]
        pal = self._theme.palette
        size = 22
        line_h = self._backend.text_size("Xg", size)[1]
        pad = 6
        box_w = max(self._backend.text_size(line, size)[0] for line in lines) + 2 * pad
        box_h = len(lines) * line_h + 2 * pad
        cx = self._offset_x + model.location[0] * self._scale
        cy = self._offset_y + model.location[1] * self._scale
        x = int(max(4, min(cx + 14, self._window_w - box_w - 4)))
        y = int(max(4, min(cy - box_h - 10, self._window_h - box_h - 4)))
        self._backend.fill_rect(
            frame, (x - 1, y - 1, box_w + 2, box_h + 2), pal.panel_line
        )
        self._backend.fill_rect(frame, (x, y, box_w, box_h), pal.panel_bg)
        for j, line in enumerate(lines):
            self._backend.draw_text(
                frame,
                line,
                (x + pad, y + pad + j * line_h + line_h // 2),
                size,
                pal.text,
                "midleft",
            )

    def close(self) -> None:
        if self._window is not None:
            pygame.display.quit()
            pygame.quit()
            self._window = None
