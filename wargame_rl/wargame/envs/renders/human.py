import math
from collections.abc import Callable

import numpy as np
import pygame

from wargame_rl.wargame.envs.env_components.distance_cache import (
    compute_levels_of_control,
)
from wargame_rl.wargame.envs.renders.renderer import Renderer
from wargame_rl.wargame.envs.wargame import WargameEnv
from wargame_rl.wargame.envs.wargame_model import WargameModel


class QuitRequested(Exception):
    """Raised when the user presses Esc to stop the application."""


class HumanRender(Renderer):
    PANEL_HEIGHT = 36  # North (title) panel
    SOUTH_PANEL_HEIGHT = 72  # South panel: 2 rows (VP row + Round/Step/Reward row)
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
        # Total window height: north panel + grid + south panel
        self._total_window_height = (
            self.GRID_SIZE + self.PANEL_HEIGHT + self.SOUTH_PANEL_HEIGHT
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

    def setup(self, env: WargameEnv) -> None:
        # Scale so board fits within GRID_SIZE on the longer side; keep square cells
        board_w = env.config.board_width
        board_h = env.config.board_height
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
            self.canvas_height + self.PANEL_HEIGHT + self.SOUTH_PANEL_HEIGHT
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

    def render(self, env: WargameEnv) -> None:
        self._process_events(env)
        if self.should_quit:
            raise QuitRequested()
        self._render_frame(env)
        while self.paused:
            self._process_events(env)
            if self.should_quit:
                raise QuitRequested()
            self._render_frame(env)
            if self.clock is not None:
                self.clock.tick(env.metadata["render_fps"])

    def _render_frame(self, env: WargameEnv) -> None:
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
                current_h - self.PANEL_HEIGHT - self.SOUTH_PANEL_HEIGHT,
            )
            self._compute_scale_and_canvas(available_w, available_h)
            self._canvas_offset_x = (current_w - self.canvas_width) // 2
            self._canvas_offset_y = (
                self.PANEL_HEIGHT + (available_h - self.canvas_height) // 2
            )

        board_width = env.config.board_width
        board_height = env.config.board_height
        wargame_models = env.wargame_models
        metadata = env.metadata
        deployment_zone = env.deployment_zone
        opponent_deployment_zone = env.opponent_deployment_zone

        # Clear window and canvas (window fill clears letterboxing after resize)
        self.window.fill((45, 45, 48))
        self.canvas.fill((255, 255, 255))

        self._draw_deployment_zone(self.canvas, deployment_zone)
        self._draw_deployment_zone_text(self.canvas, deployment_zone, "Deployment Zone")

        self._draw_deployment_zone(
            self.canvas, opponent_deployment_zone, color=(220, 200, 200)
        )
        self._draw_deployment_zone_text(
            self.canvas, opponent_deployment_zone, "Opponent Zone"
        )

        # We draw the target (with control-based coloring when env has control range)
        self._draw_target(self.canvas, env)

        # Draw movement arrows (previous -> current location)
        self._draw_movement_arrows(self.canvas, wargame_models)

        # Draw opponent movement arrows and models
        if env.opponent_models:
            self._draw_opponent_movement_arrows(self.canvas, env.opponent_models)
            self._draw_opponent_models(self.canvas, env.opponent_models)

        # Now we draw the player agent models
        self._draw_agent(self.canvas, wargame_models)

        # Finally, add some gridlines
        self._draw_gridlines(self.canvas, board_width, board_height)

        # Copy game canvas to window (centered in grid area when resized)
        self.window.blit(
            self.canvas,
            (self._canvas_offset_x, self._canvas_offset_y),
        )
        self._draw_north_panel(env)
        self._draw_south_panel(env)
        # Show tooltip for pinned model (follows model) or hovered model
        tooltip_index = (
            self._pinned_model_index
            if self._pinned_model_index is not None
            else self._get_hovered_model_index(env)
        )
        if tooltip_index is not None:
            self._draw_model_tooltip(env, tooltip_index)
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

    def _draw_north_panel(self, env: WargameEnv) -> None:
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
        menu_text = "Space: Pause | Esc: Quit"
        if self.paused:
            menu_text = "PAUSED - Space: Resume | Esc: Quit"
        text_surface = font.render(menu_text, True, (220, 220, 220))
        text_rect = text_surface.get_rect(
            center=(window_w // 2, self.PANEL_HEIGHT // 2)
        )
        self.window.blit(text_surface, text_rect)

    def _draw_south_panel(self, env: WargameEnv) -> None:
        """Draw the south panel (2 rows): row 1 = VP; row 2 = Round, Step, Reward.

        Uses a single font size and shared column anchors so both rows align
        (left column, center, right column).
        """
        if self.window is None:
            return
        window_w = self.window.get_width()
        window_h = self.window.get_height()
        panel_y = window_h - self.SOUTH_PANEL_HEIGHT
        panel_rect = pygame.Rect(0, panel_y, window_w, self.SOUTH_PANEL_HEIGHT)
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
        margin = max(12, int(window_w * 0.02))
        col_left = margin
        col_center = window_w // 2
        col_right = window_w - margin

        reward_str = f"{env.last_reward:.3f}" if env.last_reward is not None else "—"
        clock_state = env._game_clock.state
        phase_label = clock_state.phase.value.title() if clock_state.phase else "—"
        round_num = clock_state.battle_round or 0
        n_rounds = env._game_clock.n_rounds
        turn_text = f"Round: {round_num} / {n_rounds}  |  {phase_label}"
        steps_text = f"Step: {env.current_turn}"
        reward_text = f"Reward: {reward_str}"
        player_vp = getattr(env, "player_vp", 0)
        opponent_vp = getattr(env, "opponent_vp", 0)
        vp_player_text = f"Player VP: {player_vp}"
        vp_opponent_text = f"Opponent VP: {opponent_vp}"
        vp_gain_player = getattr(env, "vp_gained_this_step_player", 0)
        vp_gain_opponent = getattr(env, "vp_gained_this_step_opponent", 0)

        row1_y = panel_y + self.SOUTH_PANEL_HEIGHT // 4
        row2_y = panel_y + 3 * self.SOUTH_PANEL_HEIGHT // 4

        # Row 1: VP left-aligned and right-aligned to match row 2 columns
        vp_p_surface = font.render(vp_player_text, True, self._OBJECTIVE_PLAYER_COLOR)
        vp_o_surface = font.render(
            vp_opponent_text, True, self._OBJECTIVE_OPPONENT_COLOR
        )
        vp_p_rect = vp_p_surface.get_rect(midleft=(col_left, row1_y))
        vp_o_rect = vp_o_surface.get_rect(midright=(col_right, row1_y))
        self.window.blit(vp_p_surface, vp_p_rect)
        self.window.blit(vp_o_surface, vp_o_rect)

        # Row 2: Round (left), Step (center), Reward (right); same font and anchors
        if self.epoch is not None:
            turn_text = f"Epoch: {self.epoch}  |  {turn_text}"
        turn_surface = font.render(turn_text, True, text_color)
        steps_surface = font.render(steps_text, True, text_color)
        reward_surface = font.render(reward_text, True, text_color)
        turn_rect = turn_surface.get_rect(midleft=(col_left, row2_y))
        steps_rect = steps_surface.get_rect(center=(col_center, row2_y))
        reward_rect = reward_surface.get_rect(midright=(col_right, row2_y))
        self.window.blit(turn_surface, turn_rect)
        self.window.blit(steps_surface, steps_rect)
        self.window.blit(reward_surface, reward_rect)

        # VP-increase feedback: "+N VP" on row 1, same font for consistency
        if vp_gain_player > 0 or vp_gain_opponent > 0:
            if vp_gain_player > 0:
                gain_text = f"+{vp_gain_player} VP"
                gain_surface = font.render(gain_text, True, (255, 255, 100))
                gain_rect = gain_surface.get_rect(midleft=(vp_p_rect.right + 8, row1_y))
                self.window.blit(gain_surface, gain_rect)
            if vp_gain_opponent > 0:
                gain_text = f"+{vp_gain_opponent} VP"
                gain_surface = font.render(gain_text, True, (255, 180, 100))
                gain_rect = gain_surface.get_rect(midright=(vp_o_rect.left - 8, row1_y))
                self.window.blit(gain_surface, gain_rect)

    def _get_model_index_at(self, env: WargameEnv, mx: int, my: int) -> int | None:
        """Return the index of the wargame model at window position (mx, my), or None."""
        if not (
            self._canvas_offset_x <= mx < self._canvas_offset_x + self.canvas_width
            and self._canvas_offset_y <= my < self._canvas_offset_y + self.canvas_height
        ):
            return None
        canvas_x = float(mx - self._canvas_offset_x)
        canvas_y = float(my - self._canvas_offset_y)
        hit_radius = max(self.pix_square_size / 2, 12.0)
        for i, model in enumerate(env.wargame_models):
            center_x = (model.location[0] + 0.5) * self.pix_square_size
            center_y = (model.location[1] + 0.5) * self.pix_square_size
            dist_sq = (canvas_x - center_x) ** 2 + (canvas_y - center_y) ** 2
            if dist_sq <= hit_radius**2:
                return i
        return None

    def _get_hovered_model_index(self, env: WargameEnv) -> int | None:
        """Return the index of the wargame model under the mouse, or None."""
        mx, my = pygame.mouse.get_pos()
        return self._get_model_index_at(env, mx, my)

    def _draw_model_tooltip(self, env: WargameEnv, model_index: int) -> None:
        """Draw a popup overlay with model info near the hovered model."""
        if self.window is None:
            return
        model = env.wargame_models[model_index]
        # Model center in window coords (canvas may be offset when window is resized)
        center_x = (
            self._canvas_offset_x + (model.location[0] + 0.5) * self.pix_square_size
        )
        center_y = (
            self._canvas_offset_y + (model.location[1] + 0.5) * self.pix_square_size
        )
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

    def _process_events(self, env: WargameEnv) -> None:
        """Process pygame events for pause (Space), quit (Esc), resize, and click-to-pin tooltip."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.should_quit = True
            elif event.type == pygame.VIDEORESIZE:
                if self.window is None:
                    continue
                new_w = max(1, event.w)
                new_h = max(
                    self.PANEL_HEIGHT + self.SOUTH_PANEL_HEIGHT + 1,
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
                available_h = new_h - self.PANEL_HEIGHT - self.SOUTH_PANEL_HEIGHT
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
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # left click
                    model_index = self._get_model_index_at(
                        env, event.pos[0], event.pos[1]
                    )
                    self._pinned_model_index = model_index

    def _draw_deployment_zone(
        self,
        canvas: pygame.Surface,
        deployment_zone: np.ndarray,
        color: tuple[int, int, int] = (200, 200, 200),
    ) -> None:
        """Draw deployment zone on the canvas."""

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

    # Control colors: player-controlled, opponent-controlled, contested
    _OBJECTIVE_PLAYER_COLOR = (60, 180, 80)  # green
    _OBJECTIVE_OPPONENT_COLOR = (220, 80, 60)  # red
    _OBJECTIVE_CONTESTED_COLOR = (255, 220, 0)  # yellow

    def _draw_target(self, canvas: pygame.Surface, env: WargameEnv) -> None:
        """Draw objectives on the canvas; color by control (player/opponent/contested)."""
        objectives = env.objectives
        control_range = getattr(env.config, "objective_control_range", None)
        player_loc: np.ndarray | None = None
        opponent_loc: np.ndarray | None = None
        if control_range is not None and control_range > 0 and env.opponent_models:
            player_loc, opponent_loc = compute_levels_of_control(
                env.wargame_models,
                env.opponent_models,
                objectives,
                control_range,
            )

        for i, objective in enumerate(objectives):
            if player_loc is not None and opponent_loc is not None:
                pl, ol = float(player_loc[i]), float(opponent_loc[i])
                if pl > ol:
                    fill_color = self._OBJECTIVE_PLAYER_COLOR
                elif ol > pl:
                    fill_color = self._OBJECTIVE_OPPONENT_COLOR
                else:
                    fill_color = self._OBJECTIVE_CONTESTED_COLOR
            else:
                fill_color = (255, 100, 100)

            cx = float(objective.location[0] + 0.5) * self.pix_square_size
            cy = float(objective.location[1] + 0.5) * self.pix_square_size
            r = float(objective.radius_size * self.pix_square_size)
            pygame.draw.circle(canvas, fill_color, (cx, cy), r)
            pygame.draw.rect(
                canvas,
                (200, 80, 80),
                pygame.Rect(
                    float(objective.location[0] * self.pix_square_size),
                    float(objective.location[1] * self.pix_square_size),
                    float(self.pix_square_size),
                    float(self.pix_square_size),
                ),
                1,
            )

    def _color_for_group(self, group_id: int) -> tuple[int, int, int]:
        """Return a distinct color for the given group_id (1-based). Cycles through palette if needed."""
        index = group_id % len(self._GROUP_COLORS)
        return self._GROUP_COLORS[index]

    def _opponent_color_for_group(self, group_id: int) -> tuple[int, int, int]:
        """Return a distinct opponent color for the given group_id."""
        index = group_id % len(self._OPPONENT_COLORS)
        return self._OPPONENT_COLORS[index]

    def _draw_agent(
        self, canvas: pygame.Surface, wargame_models: list[WargameModel]
    ) -> None:
        """Draw wargame models (agents) on the canvas. Color is determined by model group_id."""
        for model in wargame_models:
            color = self._color_for_group(model.group_id)
            pygame.draw.circle(
                canvas,
                color,
                (
                    float(model.location[0] + 0.5) * self.pix_square_size,
                    float(model.location[1] + 0.5) * self.pix_square_size,
                ),
                self.pix_square_size / 3,
            )

    def _draw_opponent_models(
        self, canvas: pygame.Surface, opponent_models: list[WargameModel]
    ) -> None:
        """Draw opponent models as downward-pointing triangles."""
        for model in opponent_models:
            color = self._opponent_color_for_group(model.group_id)
            cx = float(model.location[0] + 0.5) * self.pix_square_size
            cy = float(model.location[1] + 0.5) * self.pix_square_size
            r = self.pix_square_size / 3
            # Equilateral triangle pointing down
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
            if model.previous_location is None:
                continue
            prev = model.previous_location
            curr = model.location
            if (prev == curr).all():
                continue

            color = color_for_group(model.group_id)
            faded = tuple(c + (255 - c) // 2 for c in color)

            prev_px = (
                float(prev[0] + 0.5) * self.pix_square_size,
                float(prev[1] + 0.5) * self.pix_square_size,
            )
            curr_px = (
                float(curr[0] + 0.5) * self.pix_square_size,
                float(curr[1] + 0.5) * self.pix_square_size,
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
