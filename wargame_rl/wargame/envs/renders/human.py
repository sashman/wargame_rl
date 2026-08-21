import math
from collections.abc import Callable

import numpy as np
import pygame

from wargame_rl.wargame.envs.domain.battle_view import BattleView
from wargame_rl.wargame.envs.domain.entities import alive_mask_for
from wargame_rl.wargame.envs.env_components.distance_cache import (
    compute_distances,
    objective_ownership_from_norms_offset,
)
from wargame_rl.wargame.envs.renders.renderer import Renderer
from wargame_rl.wargame.envs.types.geometry import Polygon
from wargame_rl.wargame.envs.wargame_model import WargameModel
from wargame_rl.wargame.envs.wargame_objective import WargameObjective


def los_line_color(
    view: BattleView, x0: float, y0: float, x1: float, y1: float
) -> tuple[int, int, int]:
    """Return green if LOS clear, red if blocked."""
    if view.has_line_of_sight_between_points(x0, y0, x1, y1):
        return (80, 200, 80)
    return (255, 80, 80)


class QuitRequested(Exception):
    """Raised when the user presses Esc to stop the application."""


class HumanRender(Renderer):
    PANEL_HEIGHT = 36
    SOUTH_PANEL_ROWS = 2
    GRID_SIZE = 1024  # Max width or height of the game grid in pixels

    # Distinct colors per group for player models
    _GROUP_COLORS: list[tuple[int, int, int]] = [
        (0, 0, 255),  # blue – group 1
        (60, 180, 80),  # green – group 2
        (255, 180, 0),  # orange – group 3
        (180, 80, 220),  # purple – group 4
        (0, 200, 200),  # cyan – group 5
        (200, 100, 0),  # brown – group 6
        (220, 100, 180),  # pink – group 7
        (220, 80, 60),  # red – group 8
    ]

    # Warm/red palette for opponent models
    _OPPONENT_COLORS: list[tuple[int, int, int]] = [
        (200, 40, 40),  # dark red
        (220, 100, 30),  # burnt orange
        (180, 30, 80),  # crimson
        (160, 60, 140),  # magenta
        (200, 80, 80),  # rose
        (180, 100, 20),  # amber
        (140, 30, 50),  # maroon
        (210, 60, 100),  # hot pink
    ]

    def __init__(self) -> None:
        self.window: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.canvas_width = self.GRID_SIZE
        self.canvas_height = self.GRID_SIZE
        self.canvas: pygame.Surface | None = None
        self.paused = False
        self.should_quit = False
        # Model index for tooltip pinned by click; None = show only on hover
        self._pinned_model_index: int | None = None
        # Optional epoch number to show in south panel (e.g. when recording)
        self.epoch: int | None = None
        # Total window height: north panel + grid + south panel (two rows)
        self._total_window_height = (
            self.GRID_SIZE
            + self.PANEL_HEIGHT
            + self.SOUTH_PANEL_ROWS * self.PANEL_HEIGHT
        )
        # Board dimensions (set in setup) for recomputing scale on window resize
        self._board_width: int = 50
        self._board_height: int = 50
        # Offset of canvas within window (for centered grid when resized)
        self._canvas_offset_x: int = 0
        self._canvas_offset_y: int = self.PANEL_HEIGHT
        # Last window size we used for layout (to detect resize even without VIDEORESIZE)
        self._last_window_w: int = 0
        self._last_window_h: int = 0
        # Toggle with L: draw Bresenham LOS between first alive player and opponent.
        self._debug_los: bool = False

    def _compute_scale_and_canvas(
        self, available_width: int, available_height: int
    ) -> None:
        """Set scale and canvas size to fit board in available area; keep square cells."""
        if available_width <= 0 or available_height <= 0:
            return
        scale = min(
            available_width / self._board_width,
            available_height / self._board_height,
        )
        self.pix_square_size = scale
        self.canvas_width = math.ceil(scale * self._board_width)
        self.canvas_height = math.ceil(scale * self._board_height)
        self.canvas = pygame.Surface((self.canvas_width, self.canvas_height))

    def setup(self, view: BattleView) -> None:
        # Scale so board fits within GRID_SIZE on the longer side; keep square cells
        board_w = view.config.board_width
        board_h = view.config.board_height
        self._board_width = board_w
        self._board_height = board_h

        scale = min(
            self.GRID_SIZE / board_w,
            self.GRID_SIZE / board_h,
        )
        # Use ceil so the full grid fits (no clipping of last row/column)
        self.canvas_width = math.ceil(scale * board_w)
        self.canvas_height = math.ceil(scale * board_h)
        self.pix_square_size = scale
        self._total_window_height = (
            self.canvas_height
            + self.PANEL_HEIGHT
            + self.SOUTH_PANEL_ROWS * self.PANEL_HEIGHT
        )
        self._canvas_offset_x = 0
        self._canvas_offset_y = self.PANEL_HEIGHT

        if self.window is None:
            pygame.init()
            pygame.display.init()
            size = (self.canvas_width, self._total_window_height)
            self.window = pygame.display.set_mode(size, pygame.RESIZABLE)
            pygame.display.set_caption("Wargame")
            self._last_window_w, self._last_window_h = size
        else:
            current = self.window.get_size()
            if current != (self.canvas_width, self._total_window_height):
                self.window = pygame.display.set_mode(
                    (self.canvas_width, self._total_window_height),
                    pygame.RESIZABLE,
                )
                self._last_window_w, self._last_window_h = self.window.get_size()

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.canvas is None or self.canvas.get_size() != (
            self.canvas_width,
            self.canvas_height,
        ):
            self.canvas = pygame.Surface((self.canvas_width, self.canvas_height))
        self.canvas.fill((255, 255, 255))
        if self.window is not None:
            self._last_window_w, self._last_window_h = self.window.get_size()

    def render(self, view: BattleView) -> None:
        self._process_events(view)
        if self.should_quit:
            raise QuitRequested()
        self._render_frame(view)
        while self.paused:
            self._process_events(view)
            if self.should_quit:
                raise QuitRequested()
            self._render_frame(view)
            if self.clock is not None:
                self.clock.tick(view.metadata["render_fps"])

    def _render_frame(self, view: BattleView) -> None:
        if self.canvas is None:
            raise ValueError("Canvas is not initialized")
        if self.window is None:
            raise ValueError("Window is not initialized")
        if self.clock is None:
            raise ValueError("Clock is not initialized")

        # When window size changes (expand/shrink), scale grid to fill the new size.
        # This handles both VIDEORESIZE and platforms where resize is detected via get_size().
        current_w, current_h = self.window.get_size()
        if (current_w, current_h) != (self._last_window_w, self._last_window_h):
            self._last_window_w, self._last_window_h = current_w, current_h
            self._total_window_height = current_h
            available_w = current_w
            available_h = max(
                1,
                current_h
                - self.PANEL_HEIGHT
                - self.SOUTH_PANEL_ROWS * self.PANEL_HEIGHT,
            )
            self._compute_scale_and_canvas(available_w, available_h)
            self._canvas_offset_x = (current_w - self.canvas_width) // 2
            self._canvas_offset_y = (
                self.PANEL_HEIGHT + (available_h - self.canvas_height) // 2
            )

        board_width = view.config.board_width
        board_height = view.config.board_height
        objectives = view.objectives
        wargame_models = view.player_models
        metadata = view.metadata
        deployment_zone = view.deployment_zone
        opponent_deployment_zone = view.opponent_deployment_zone

        # Clear window and canvas (window fill clears letterboxing after resize)
        self.window.fill((45, 45, 48))
        self.canvas.fill((255, 255, 255))

        # A map's own zone is a polygon on five of the six real deployments, so
        # the rectangle is only correct when the map has no outline of its own.
        self._draw_deployment_zone(
            self.canvas, deployment_zone, outline=view.deployment_outline
        )
        self._draw_deployment_zone_text(self.canvas, deployment_zone, "Deployment Zone")

        self._draw_deployment_zone(
            self.canvas,
            opponent_deployment_zone,
            color=(220, 200, 200),
            outline=view.opponent_deployment_outline,
        )
        self._draw_deployment_zone_text(
            self.canvas, opponent_deployment_zone, "Opponent Zone"
        )

        self._draw_terrain(self.canvas, view)

        # We draw the target
        self._draw_target(self.canvas, wargame_models, view.opponent_models, objectives)

        # Draw movement arrows (previous -> current location)
        self._draw_movement_arrows(self.canvas, wargame_models)

        # Draw opponent movement arrows and models
        if view.opponent_models:
            self._draw_opponent_movement_arrows(self.canvas, view.opponent_models)
            self._draw_opponent_models(self.canvas, view.opponent_models)

        # Now we draw the player agent models
        self._draw_agent(self.canvas, wargame_models)

        if self._debug_los:
            self._draw_debug_los_line(view)

        # Finally, add some gridlines
        self._draw_gridlines(self.canvas, board_width, board_height)

        # Copy game canvas to window (centered in grid area when resized)
        self.window.blit(
            self.canvas,
            (self._canvas_offset_x, self._canvas_offset_y),
        )
        self._draw_north_panel(view)
        self._draw_south_panel(view)
        # Show tooltip for pinned model (follows model) or hovered model
        tooltip_index = (
            self._pinned_model_index
            if self._pinned_model_index is not None
            else self._get_hovered_model_index(view)
        )
        if tooltip_index is not None:
            self._draw_model_tooltip(view, tooltip_index)
        pygame.event.pump()
        pygame.display.update()

        self.clock.tick(metadata["render_fps"])

    def get_frame_array(self) -> np.ndarray:
        """Return the current window as RGB array (height, width, 3) for video recording."""
        if self.window is None:
            raise ValueError("Window is not initialized")
        # pygame.surfarray.array3d returns (width, height, 3); we need (height, width, 3)
        return np.asarray(
            np.transpose(pygame.surfarray.array3d(self.window), (1, 0, 2)),
            dtype=np.uint8,
        )

    def _draw_north_panel(self, view: BattleView) -> None:
        """Draw the north panel with hot key menu."""
        if self.window is None:
            return
        window_w = self.window.get_width()
        panel_rect = pygame.Rect(0, 0, window_w, self.PANEL_HEIGHT)
        pygame.draw.rect(self.window, (45, 45, 48), panel_rect)
        pygame.draw.line(
            self.window,
            (80, 80, 84),
            (0, self.PANEL_HEIGHT),
            (window_w, self.PANEL_HEIGHT),
            width=1,
        )
        font = pygame.font.Font(None, 24)
        menu_text = "Space: Pause | Esc: Quit | L: LOS debug"
        if self.paused:
            menu_text = "PAUSED - Space: Resume | Esc: Quit | L: LOS debug"
        text_surface = font.render(menu_text, True, (220, 220, 220))
        text_rect = text_surface.get_rect(
            center=(window_w // 2, self.PANEL_HEIGHT // 2)
        )
        self.window.blit(text_surface, text_rect)

    def _draw_south_panel(self, view: BattleView) -> None:
        """Draw the south panel with environment information (two rows)."""
        if self.window is None:
            return
        window_w = self.window.get_width()
        window_h = self.window.get_height()
        south_h = self.SOUTH_PANEL_ROWS * self.PANEL_HEIGHT
        panel_y = window_h - south_h
        panel_rect = pygame.Rect(0, panel_y, window_w, south_h)
        pygame.draw.rect(self.window, (45, 45, 48), panel_rect)
        pygame.draw.line(
            self.window,
            (80, 80, 84),
            (0, panel_y),
            (window_w, panel_y),
            width=1,
        )
        font = pygame.font.Font(None, 24)
        text_color = (220, 220, 220)
        reward_str = f"{view.last_reward:.3f}" if view.last_reward is not None else "—"
        clock_state = view.game_clock_state
        phase_label = clock_state.phase.value.title() if clock_state.phase else "—"
        round_num = clock_state.battle_round or 0
        n_rounds = view.n_rounds
        turn_text = f"Round: {round_num} / {n_rounds}  |  {phase_label}"
        steps_text = f"Step: {view.current_turn}"
        reward_text = f"Reward: {reward_str}"
        center_y_row1 = panel_y + self.PANEL_HEIGHT // 2
        if self.epoch is not None:
            epoch_text = f"Epoch: {self.epoch}"
            epoch_surface = font.render(epoch_text, True, text_color)
            turn_surface = font.render(turn_text, True, text_color)
            steps_surface = font.render(steps_text, True, text_color)
            reward_surface = font.render(reward_text, True, text_color)
            epoch_rect = epoch_surface.get_rect(center=(window_w // 8, center_y_row1))
            turn_rect = turn_surface.get_rect(center=(3 * window_w // 8, center_y_row1))
            steps_rect = steps_surface.get_rect(
                center=(5 * window_w // 8, center_y_row1)
            )
            reward_rect = reward_surface.get_rect(
                center=(7 * window_w // 8, center_y_row1)
            )
            self.window.blit(epoch_surface, epoch_rect)
        else:
            turn_surface = font.render(turn_text, True, text_color)
            steps_surface = font.render(steps_text, True, text_color)
            reward_surface = font.render(reward_text, True, text_color)
            turn_rect = turn_surface.get_rect(center=(window_w // 6, center_y_row1))
            steps_rect = steps_surface.get_rect(center=(window_w // 2, center_y_row1))
            reward_rect = reward_surface.get_rect(
                center=(5 * window_w // 6, center_y_row1)
            )
        self.window.blit(turn_surface, turn_rect)
        self.window.blit(steps_surface, steps_rect)
        self.window.blit(reward_surface, reward_rect)

        # Row 2: victory points and delta this step
        center_y_row2 = panel_y + self.PANEL_HEIGHT + self.PANEL_HEIGHT // 2
        self._draw_vp_readout(
            font,
            "Player VP:",
            view.player_vp,
            view.player_vp_delta,
            window_w // 4,
            center_y_row2,
            text_color,
        )
        self._draw_vp_readout(
            font,
            "Opponent VP:",
            view.opponent_vp,
            view.opponent_vp_delta,
            3 * window_w // 4,
            center_y_row2,
            text_color,
        )

    def _draw_vp_readout(
        self,
        font: "pygame.font.Font",
        label: str,
        value: int,
        delta: int,
        center_x: int,
        center_y: int,
        color: tuple[int, int, int],
    ) -> None:
        """Draw "<label> <value> (+delta)" with every part at a fixed position.

        The delta appears for one step every few rounds. Rendering the line as a
        single centred string made the label and the number jump sideways each
        time it did, which is unreadable in a recording. So the layout reserves
        room for the widest number and the widest delta whether or not they are
        drawn: the label is left-anchored, the number is right-anchored inside a
        fixed field so it grows leftward as it gains digits, and the delta is
        left-anchored past that field. Nothing moves when the delta appears.
        """
        if self.window is None:
            return
        gap = font.size(" ")[0]
        value_field = font.size("000")[0]
        delta_field = font.size("(+00)")[0]

        label_surface = font.render(label, True, color)
        value_surface = font.render(str(value), True, color)

        total_width = label_surface.get_width() + gap + value_field + gap + delta_field
        left = center_x - total_width // 2

        self.window.blit(
            label_surface, label_surface.get_rect(midleft=(left, center_y))
        )
        value_right = left + label_surface.get_width() + gap + value_field
        self.window.blit(
            value_surface, value_surface.get_rect(midright=(value_right, center_y))
        )
        if delta > 0:
            delta_surface = font.render(f"(+{delta})", True, color)
            self.window.blit(
                delta_surface,
                delta_surface.get_rect(midleft=(value_right + gap, center_y)),
            )

    def _get_model_index_at(self, view: BattleView, mx: int, my: int) -> int | None:
        """Return the index of the wargame model at window position (mx, my), or None."""
        if not (
            self._canvas_offset_x <= mx < self._canvas_offset_x + self.canvas_width
            and self._canvas_offset_y <= my < self._canvas_offset_y + self.canvas_height
        ):
            return None
        canvas_x = float(mx - self._canvas_offset_x)
        canvas_y = float(my - self._canvas_offset_y)
        hit_radius = max(self.pix_square_size / 2, 12.0)
        for i, model in enumerate(view.player_models):
            center_x = model.location[0] * self.pix_square_size
            center_y = model.location[1] * self.pix_square_size
            dist_sq = (canvas_x - center_x) ** 2 + (canvas_y - center_y) ** 2
            if dist_sq <= hit_radius**2:
                return i
        return None

    def _get_hovered_model_index(self, view: BattleView) -> int | None:
        """Return the index of the wargame model under the mouse, or None."""
        mx, my = pygame.mouse.get_pos()
        return self._get_model_index_at(view, mx, my)

    def _draw_model_tooltip(self, view: BattleView, model_index: int) -> None:
        """Draw a popup overlay with model info near the hovered model."""
        if self.window is None:
            return
        model = view.player_models[model_index]
        # Model center in window coords (canvas may be offset when window is resized)
        center_x = self._canvas_offset_x + model.location[0] * self.pix_square_size
        center_y = self._canvas_offset_y + model.location[1] * self.pix_square_size
        latest = (
            model.model_rewards_history[-1] if model.model_rewards_history else None
        )
        if latest is not None:
            lines = [
                f"Location: ({model.location[0]}, {model.location[1]})",
                f"Group ID: {model.group_id}",
                f"Closest objective reward: {latest.closest_objective_reward:.3f}",
                f"Group distance violation penalty: {latest.group_distance_violation_penalty:.3f}",
                f"Total reward: {latest.total_reward:.3f}",
            ]
        else:
            lines = [
                f"Location: ({model.location[0]}, {model.location[1]})",
                f"Group ID: {model.group_id}",
                "Closest objective reward: —",
            ]
        font = pygame.font.Font(None, 22)
        padding = 6
        line_height = font.get_height()
        text_color = (220, 220, 220)
        bg_color = (45, 45, 48)
        border_color = (80, 80, 84)
        max_w = 0
        surfaces = []
        for line in lines:
            s = font.render(line, True, text_color)
            surfaces.append(s)
            max_w = max(max_w, s.get_width())
        box_w = max_w + 2 * padding
        box_h = len(lines) * line_height + 2 * padding
        # Position above and slightly right of model, keep on screen
        window_w = self.window.get_width()
        window_h = self.window.get_height()
        tooltip_x = center_x + 14
        tooltip_y = center_y - box_h - 10
        tooltip_x = max(4, min(tooltip_x, window_w - box_w - 4))
        tooltip_y = max(4, min(tooltip_y, window_h - box_h - 4))
        rect = pygame.Rect(tooltip_x, tooltip_y, box_w, box_h)
        pygame.draw.rect(self.window, border_color, rect.inflate(2, 2))
        pygame.draw.rect(self.window, bg_color, rect)
        for j, s in enumerate(surfaces):
            self.window.blit(s, (rect.x + padding, rect.y + padding + j * line_height))

    def _draw_debug_los_line(self, view: BattleView) -> None:
        """Draw the sight line from the first alive player model to the first opponent.

        A straight segment, because that is now literally what is traced: the
        board is continuous and sight is sampled along the line rather than
        walked cell by cell, so a stepped polyline would draw a ray the domain
        no longer casts. Colour reflects the verdict: green if clear, red if
        blocked.
        """
        if self.canvas is None:
            return
        player_alive = alive_mask_for(view.player_models)
        try:
            p_idx = next(i for i, ok in enumerate(player_alive) if ok)
        except StopIteration:
            return
        if not view.opponent_models:
            return
        opp_alive = alive_mask_for(view.opponent_models)
        try:
            o_idx = next(i for i, ok in enumerate(opp_alive) if ok)
        except StopIteration:
            return
        pm = view.player_models[p_idx]
        om = view.opponent_models[o_idx]
        x0, y0 = float(pm.location[0]), float(pm.location[1])
        x1, y1 = float(om.location[0]), float(om.location[1])
        color = los_line_color(view, x0, y0, x1, y1)
        w = max(1, int(self.pix_square_size * 0.08))
        pygame.draw.line(
            self.canvas,
            color,
            (x0 * self.pix_square_size, y0 * self.pix_square_size),
            (x1 * self.pix_square_size, y1 * self.pix_square_size),
            width=w,
        )

    def _process_events(self, view: BattleView) -> None:
        """Process pygame events for pause (Space), quit (Esc), resize, and click-to-pin tooltip."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.should_quit = True
            elif event.type == pygame.VIDEORESIZE:
                if self.window is None:
                    continue
                new_w = max(1, event.w)
                new_h = max(
                    self.PANEL_HEIGHT + self.SOUTH_PANEL_ROWS * self.PANEL_HEIGHT + 1,
                    event.h,
                )
                # Only resize when size actually changed to avoid feedback loop
                # (set_mode can trigger another VIDEORESIZE on some systems)
                current_w, current_h = self.window.get_size()
                if (new_w, new_h) == (current_w, current_h):
                    continue
                self.window = pygame.display.set_mode(
                    (new_w, new_h),
                    pygame.RESIZABLE,
                )
                self._last_window_w, self._last_window_h = new_w, new_h
                self._total_window_height = new_h
                available_w = new_w
                available_h = (
                    new_h
                    - self.PANEL_HEIGHT
                    - self.SOUTH_PANEL_ROWS * self.PANEL_HEIGHT
                )
                self._compute_scale_and_canvas(available_w, available_h)
                self._canvas_offset_x = (new_w - self.canvas_width) // 2
                self._canvas_offset_y = (
                    self.PANEL_HEIGHT + (available_h - self.canvas_height) // 2
                )
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_ESCAPE:
                    self.should_quit = True
                elif event.key == pygame.K_l:
                    self._debug_los = not self._debug_los
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # left click
                    model_index = self._get_model_index_at(
                        view, event.pos[0], event.pos[1]
                    )
                    self._pinned_model_index = model_index

    def _draw_deployment_zone(
        self,
        canvas: pygame.Surface,
        deployment_zone: np.ndarray,
        color: tuple[int, int, int] = (200, 200, 200),
        outline: Polygon | None = None,
    ) -> None:
        """Draw deployment zone on the canvas, as its outline where it has one."""

        if outline is not None:
            pygame.draw.polygon(
                canvas,
                color,
                [
                    (float(vx) * self.pix_square_size, float(vy) * self.pix_square_size)
                    for vx, vy in outline.vertices
                ],
            )
            return

        x = float(deployment_zone[0] * self.pix_square_size)
        y = float(deployment_zone[1] * self.pix_square_size)
        width = float((deployment_zone[2] - deployment_zone[0]) * self.pix_square_size)
        height = float((deployment_zone[3] - deployment_zone[1]) * self.pix_square_size)

        pygame.draw.rect(
            canvas,
            color,
            pygame.Rect(x, y, width, height),
        )

    def _draw_deployment_zone_text(
        self, canvas: pygame.Surface, deployment_zone: np.ndarray, label: str = ""
    ) -> None:
        """Draw a label centred inside the deployment zone."""
        font = pygame.font.Font(None, 48)
        text = font.render(label or "Deployment Zone", True, (240, 240, 240))
        text_width, text_height = text.get_size()

        # Calculate the center position
        zone_width = deployment_zone[2] - deployment_zone[0]

        zone_height = deployment_zone[3] - deployment_zone[1]
        center_x = deployment_zone[0] + zone_width / 2
        center_y = deployment_zone[1] + zone_height / 2

        # Ensure text is contained within the deployment zone
        text_x = max(center_x, deployment_zone[0])
        text_y = max(center_y, deployment_zone[1])

        canvas.blit(
            text,
            (
                text_x * self.pix_square_size - text_width / 2,
                text_y * self.pix_square_size - text_height / 2,
            ),
        )

    def _to_pixels(self, vertices: np.ndarray) -> list[tuple[float, float]]:
        """Board-unit vertices to canvas pixels.

        No `+ 0.5` anywhere. That offset moved a *cell index* to the cell's
        centre; a continuous coordinate is already the point, and adding half a
        unit would draw every model and every ruin corner shifted down and right
        of where the rules put it.
        """
        return [
            (float(x) * self.pix_square_size, float(y) * self.pix_square_size)
            for x, y in vertices
        ]

    def _draw_terrain(self, canvas: pygame.Surface, view: BattleView) -> None:
        """Draw each terrain piece as its translucent outline, not its bounding box.

        Drawing the bounding box would show an L-shaped ruin as a solid square —
        a picture of terrain the sight trace is not using, which is exactly the
        confusion polygon terrain was introduced to end.
        """
        fill_color = (140, 120, 90)
        outline_color = (100, 80, 60)
        label_color = (60, 50, 40)
        width, height = canvas.get_size()
        # A piece that is also an objective is labelled as one. Under
        # `objectives_on_terrain` the prize and the cover are the same ground,
        # and a board where every piece says "Ruin" hides exactly the thing the
        # scenario is about.
        objective_outlines = {
            objective.area.vertices.tobytes()
            for objective in view.objectives
            if objective.area is not None
        }
        for fp in view.terrain.footprints:
            points = self._to_pixels(fp.polygon.vertices)
            # Filled through a per-piece alpha surface: pygame's polygon fill has
            # no alpha channel of its own.
            fill_surf = pygame.Surface((width, height), pygame.SRCALPHA)
            pygame.draw.polygon(fill_surf, (*fill_color, 90), points)
            canvas.blit(fill_surf, (0, 0))
            pygame.draw.polygon(canvas, outline_color, points, width=2)
            font = pygame.font.Font(None, max(16, int(self.pix_square_size * 0.8)))
            is_objective = fp.polygon.vertices.tobytes() in objective_outlines
            text = font.render(
                "OBJECTIVE" if is_objective else "Ruin", True, label_color
            )
            centre = fp.polygon.centroid * self.pix_square_size
            canvas.blit(text, text.get_rect(center=(centre[0], centre[1])))

    def _draw_target(
        self,
        canvas: pygame.Surface,
        player_models: list[WargameModel],
        opponent_models: list[WargameModel],
        objectives: list[WargameObjective],
    ) -> None:
        """Draw objectives with ownership fill and a constant grey rim."""
        if not objectives:
            return

        player_alive = alive_mask_for(player_models)
        player_cache = compute_distances(
            player_models, objectives, alive_mask=player_alive
        )
        n_obj = len(objectives)
        if opponent_models:
            opp_alive = alive_mask_for(opponent_models)
            opponent_cache = compute_distances(
                opponent_models, objectives, alive_mask=opp_alive
            )
            opponent_norms = opponent_cache.model_obj_norms_offset
        else:
            opponent_norms = np.zeros((0, n_obj), dtype=np.float64)

        player_controls, opponent_controls = objective_ownership_from_norms_offset(
            player_cache.model_obj_norms_offset,
            opponent_norms,
            player_cache.obj_radii,
        )

        base_rim_color = (90, 90, 90)  # dark grey
        player_rim_color = (120, 220, 140)  # light green
        opponent_rim_color = (255, 105, 180)  # pink

        base_width = max(2, int(round(self.pix_square_size / 8)))
        for i, objective in enumerate(objectives):
            if player_controls[i]:
                fill: tuple[int, int, int] | None = player_rim_color
            elif opponent_controls[i]:
                fill = opponent_rim_color
            else:
                fill = None

            if objective.area is not None:
                # An area objective IS its terrain piece, so it is drawn on top
                # of the footprint rather than beside it: same outline, a
                # translucent ownership wash, and a heavier rim to say the
                # ground is the prize.
                self._draw_area_objective(canvas, objective.area, fill, base_rim_color)
                continue

            cx = int(round(float(objective.location[0]) * self.pix_square_size))
            cy = int(round(float(objective.location[1]) * self.pix_square_size))
            radius_px = max(
                1,
                int(round(float(objective.radius_size) * float(self.pix_square_size))),
            )
            rim_width = min(base_width, max(1, radius_px // 2))

            # Fill first, then draw the grey rim so the rim stays constant.
            if fill is not None:
                pygame.draw.circle(canvas, fill, (cx, cy), radius_px)
            pygame.draw.circle(canvas, base_rim_color, (cx, cy), radius_px, rim_width)

    def _draw_area_objective(
        self,
        canvas: pygame.Surface,
        area: Polygon,
        fill: tuple[int, int, int] | None,
        rim_color: tuple[int, int, int],
    ) -> None:
        """Overlay an area objective on the terrain piece it occupies.

        Always washed, even when uncontrolled: a contested point that looked
        identical to a plain ruin would make the one thing the scenario is about
        invisible on the board.
        """
        points = self._to_pixels(area.vertices)
        width, height = canvas.get_size()
        wash = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.polygon(wash, (*(fill or rim_color), 120), points)
        canvas.blit(wash, (0, 0))
        pygame.draw.polygon(
            canvas, rim_color, points, width=max(3, int(self.pix_square_size / 6))
        )

    def _color_for_group(self, group_id: int) -> tuple[int, int, int]:
        """Return a distinct color for the given group_id (1-based). Cycles through palette if needed."""
        index = group_id % len(self._GROUP_COLORS)
        return self._GROUP_COLORS[index]

    def _opponent_color_for_group(self, group_id: int) -> tuple[int, int, int]:
        """Return a distinct opponent color for the given group_id."""
        index = group_id % len(self._OPPONENT_COLORS)
        return self._OPPONENT_COLORS[index]

    def _base_radius_px(self, model: WargameModel) -> float:
        """Pixel radius of a model's base.

        Drawn at the *real* radius when the model has one, so the picture shows
        the footprint the rules use — bases that cannot overlap, and objective
        range measured from the edge. A dimensionless model keeps the old
        third-of-a-cell token, which is a legibility choice and nothing else.
        """
        if model.base_radius > 0.0:
            return float(model.base_radius) * self.pix_square_size
        return self.pix_square_size / 3

    def _draw_agent(
        self, canvas: pygame.Surface, wargame_models: list[WargameModel]
    ) -> None:
        """Draw wargame models (agents) on the canvas. Color is determined by model group_id."""
        for model in wargame_models:
            if not model.is_alive:
                grey = (180, 180, 180)
                cx = float(model.location[0]) * self.pix_square_size
                cy = float(model.location[1]) * self.pix_square_size
                r = self._base_radius_px(model)
                pygame.draw.circle(canvas, grey, (cx, cy), r)
                xr = self.pix_square_size / 4
                pygame.draw.line(
                    canvas,
                    (120, 120, 120),
                    (cx - xr, cy - xr),
                    (cx + xr, cy + xr),
                    2,
                )
                pygame.draw.line(
                    canvas,
                    (120, 120, 120),
                    (cx + xr, cy - xr),
                    (cx - xr, cy + xr),
                    2,
                )
                continue
            color = self._color_for_group(model.group_id)
            centre = (
                float(model.location[0]) * self.pix_square_size,
                float(model.location[1]) * self.pix_square_size,
            )
            radius = self._base_radius_px(model)
            pygame.draw.circle(canvas, color, centre, radius)
            # A rim, so adjacent bases in a packed squad read as separate models
            # rather than as one blob -- which is the whole point of drawing the
            # real radius.
            pygame.draw.circle(canvas, (30, 30, 30), centre, radius, width=1)

    def _draw_opponent_models(
        self, canvas: pygame.Surface, opponent_models: list[WargameModel]
    ) -> None:
        """Draw opponent models as downward-pointing triangles."""
        for model in opponent_models:
            cx = float(model.location[0]) * self.pix_square_size
            cy = float(model.location[1]) * self.pix_square_size
            r = self._base_radius_px(model)
            if not model.is_alive:
                grey = (180, 180, 180)
                top_left = (cx - r, cy - r * 0.6)
                top_right = (cx + r, cy - r * 0.6)
                bottom = (cx, cy + r * 0.8)
                pygame.draw.polygon(canvas, grey, [top_left, top_right, bottom])
                xr = self.pix_square_size / 4
                pygame.draw.line(
                    canvas,
                    (120, 120, 120),
                    (cx - xr, cy - xr),
                    (cx + xr, cy + xr),
                    2,
                )
                pygame.draw.line(
                    canvas,
                    (120, 120, 120),
                    (cx + xr, cy - xr),
                    (cx - xr, cy + xr),
                    2,
                )
                continue
            color = self._opponent_color_for_group(model.group_id)
            top_left = (cx - r, cy - r * 0.6)
            top_right = (cx + r, cy - r * 0.6)
            bottom = (cx, cy + r * 0.8)
            pygame.draw.polygon(canvas, color, [top_left, top_right, bottom])

    def _draw_movement_arrows_for_models(
        self,
        canvas: pygame.Surface,
        models: list[WargameModel],
        color_for_group: Callable[[int], tuple[int, int, int]],
    ) -> None:
        """Draw a small arrow from each model's previous to current location."""
        for model in models:
            if not model.is_alive:
                continue
            if model.previous_location is None:
                continue
            prev = model.previous_location
            curr = model.location
            if (prev == curr).all():
                continue

            color = color_for_group(model.group_id)
            faded = tuple(c + (255 - c) // 2 for c in color)

            prev_px = (
                float(prev[0]) * self.pix_square_size,
                float(prev[1]) * self.pix_square_size,
            )
            curr_px = (
                float(curr[0]) * self.pix_square_size,
                float(curr[1]) * self.pix_square_size,
            )

            line_width = max(3, int(self.pix_square_size / 4))
            pygame.draw.line(canvas, faded, prev_px, curr_px, width=line_width)

            dx = curr_px[0] - prev_px[0]
            dy = curr_px[1] - prev_px[1]
            length = math.hypot(dx, dy)
            if length < 1e-6:
                continue
            ux, uy = dx / length, dy / length
            head_len = min(self.pix_square_size * 0.45, length * 0.4)
            head_w = head_len * 0.5
            tip = curr_px
            left = (
                tip[0] - ux * head_len - uy * head_w,
                tip[1] - uy * head_len + ux * head_w,
            )
            right = (
                tip[0] - ux * head_len + uy * head_w,
                tip[1] - uy * head_len - ux * head_w,
            )
            pygame.draw.polygon(canvas, faded, [tip, left, right])

    def _draw_movement_arrows(
        self, canvas: pygame.Surface, wargame_models: list[WargameModel]
    ) -> None:
        """Draw a small arrow from each model's previous location to its current location."""
        self._draw_movement_arrows_for_models(
            canvas, wargame_models, self._color_for_group
        )

    def _draw_opponent_movement_arrows(
        self, canvas: pygame.Surface, opponent_models: list[WargameModel]
    ) -> None:
        """Draw movement arrows for opponent models using opponent colors."""
        self._draw_movement_arrows_for_models(
            canvas, opponent_models, self._opponent_color_for_group
        )

    def _draw_gridlines(
        self,
        canvas: pygame.Surface,
        board_width: int,
        board_height: int,
    ) -> None:
        """Draw gridlines on the canvas. Endpoints clamped to canvas bounds."""
        max_x = float(self.canvas_width - 1)
        max_y = float(self.canvas_height - 1)
        grid_color = (210, 210, 210)
        for y in range(board_height + 1):
            py = min(self.pix_square_size * y, max_y)
            pygame.draw.line(
                canvas,
                grid_color,
                (0, py),
                (max_x, py),
                width=1,
            )
        for x in range(board_width + 1):
            px = min(self.pix_square_size * x, max_x)
            pygame.draw.line(
                canvas,
                grid_color,
                (px, 0),
                (px, max_y),
                width=1,
            )

    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
