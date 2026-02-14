import numpy as np
import pygame

from wargame_rl.wargame.envs.renders.renderer import Renderer
from wargame_rl.wargame.envs.wargame import WargameEnv, WargameObjective
from wargame_rl.wargame.envs.wargame_model import WargameModel


class QuitRequested(Exception):
    """Raised when the user presses Esc to stop the application."""


class HumanRender(Renderer):
    PANEL_HEIGHT = 36
    GRID_SIZE = 1024  # Width and height of the game grid in pixels

    def __init__(self) -> None:
        self.window: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.window_size = self.GRID_SIZE
        self.canvas: pygame.Surface | None = None
        self.paused = False
        self.should_quit = False
        # Model index for tooltip pinned by click; None = show only on hover
        self._pinned_model_index: int | None = None
        # Optional epoch number to show in south panel (e.g. when recording)
        self.epoch: int | None = None
        # Total window height: north panel + grid + south panel
        self._total_window_height = self.GRID_SIZE + 2 * self.PANEL_HEIGHT

    def setup(self, env: WargameEnv) -> None:
        if self.window is None:
            pygame.init()
            pygame.display.init()
            size = (self.window_size, self._total_window_height)
            self.window = pygame.display.set_mode(size)
            pygame.display.set_caption("Wargame")

        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.canvas = pygame.Surface((self.window_size, self.window_size))
        self.canvas.fill((255, 255, 255))
        self.pix_square_size = (
            self.window_size / env.config.size
        )  # The size of a single grid square in pixels

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

        size = env.config.size
        objectives = env.objectives
        wargame_models = env.wargame_models
        metadata = env.metadata
        deployment_zone = env.deployment_zone

        # Clear the canvas
        self.canvas.fill((255, 255, 255))

        self._draw_deployment_zone(self.canvas, deployment_zone)

        self._draw_deployment_zone_text(self.canvas, deployment_zone)

        # We draw the target
        self._draw_target(self.canvas, objectives)

        # Now we draw the agent
        self._draw_agent(self.canvas, wargame_models)

        # Finally, add some gridlines
        self._draw_gridlines(self.canvas, size)

        # Copy game canvas to window between north and south panels
        self.window.blit(self.canvas, (0, self.PANEL_HEIGHT))
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

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to
        # keep the framerate stable.
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
        panel_rect = pygame.Rect(0, 0, self.window_size, self.PANEL_HEIGHT)
        pygame.draw.rect(self.window, (45, 45, 48), panel_rect)
        pygame.draw.line(
            self.window,
            (80, 80, 84),
            (0, self.PANEL_HEIGHT),
            (self.window_size, self.PANEL_HEIGHT),
            width=1,
        )
        font = pygame.font.Font(None, 24)
        menu_text = "Space: Pause | Esc: Quit"
        if self.paused:
            menu_text = "PAUSED - Space: Resume | Esc: Quit"
        text_surface = font.render(menu_text, True, (220, 220, 220))
        text_rect = text_surface.get_rect(
            center=(self.window_size // 2, self.PANEL_HEIGHT // 2)
        )
        self.window.blit(text_surface, text_rect)

    def _draw_south_panel(self, env: WargameEnv) -> None:
        """Draw the south panel with environment information."""
        if self.window is None:
            return
        panel_y = self.PANEL_HEIGHT + self.window_size
        panel_rect = pygame.Rect(0, panel_y, self.window_size, self.PANEL_HEIGHT)
        pygame.draw.rect(self.window, (45, 45, 48), panel_rect)
        pygame.draw.line(
            self.window,
            (80, 80, 84),
            (0, panel_y),
            (self.window_size, panel_y),
            width=1,
        )
        font = pygame.font.Font(None, 24)
        text_color = (220, 220, 220)
        reward_str = f"{env.last_reward:.3f}" if env.last_reward is not None else "—"
        turn_text = f"Turn: {env.current_turn} / {env.max_turns}"
        steps_text = f"Steps: {env.current_turn}"
        reward_text = f"Reward: {reward_str}"
        center_y = panel_y + self.PANEL_HEIGHT // 2
        if self.epoch is not None:
            epoch_text = f"Epoch: {self.epoch}"
            epoch_surface = font.render(epoch_text, True, text_color)
            turn_surface = font.render(turn_text, True, text_color)
            steps_surface = font.render(steps_text, True, text_color)
            reward_surface = font.render(reward_text, True, text_color)
            epoch_rect = epoch_surface.get_rect(
                center=(self.window_size // 8, center_y)
            )
            turn_rect = turn_surface.get_rect(
                center=(3 * self.window_size // 8, center_y)
            )
            steps_rect = steps_surface.get_rect(
                center=(5 * self.window_size // 8, center_y)
            )
            reward_rect = reward_surface.get_rect(
                center=(7 * self.window_size // 8, center_y)
            )
            self.window.blit(epoch_surface, epoch_rect)
        else:
            turn_surface = font.render(turn_text, True, text_color)
            steps_surface = font.render(steps_text, True, text_color)
            reward_surface = font.render(reward_text, True, text_color)
            turn_rect = turn_surface.get_rect(center=(self.window_size // 6, center_y))
            steps_rect = steps_surface.get_rect(
                center=(self.window_size // 2, center_y)
            )
            reward_rect = reward_surface.get_rect(
                center=(5 * self.window_size // 6, center_y)
            )
        self.window.blit(turn_surface, turn_rect)
        self.window.blit(steps_surface, steps_rect)
        self.window.blit(reward_surface, reward_rect)

    def _get_model_index_at(self, env: WargameEnv, mx: int, my: int) -> int | None:
        """Return the index of the wargame model at window position (mx, my), or None."""
        if not (
            0 <= mx < self.window_size
            and self.PANEL_HEIGHT <= my < self.PANEL_HEIGHT + self.window_size
        ):
            return None
        canvas_x = float(mx)
        canvas_y = float(my - self.PANEL_HEIGHT)
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
        # Model center in window coords
        center_x = (model.location[0] + 0.5) * self.pix_square_size
        center_y = self.PANEL_HEIGHT + (model.location[1] + 0.5) * self.pix_square_size
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
        tooltip_x = center_x + 14
        tooltip_y = center_y - box_h - 10
        tooltip_x = max(4, min(tooltip_x, self.window_size - box_w - 4))
        tooltip_y = max(
            self.PANEL_HEIGHT + 4, min(tooltip_y, self._total_window_height - box_h - 4)
        )
        rect = pygame.Rect(tooltip_x, tooltip_y, box_w, box_h)
        pygame.draw.rect(self.window, border_color, rect.inflate(2, 2))
        pygame.draw.rect(self.window, bg_color, rect)
        for j, s in enumerate(surfaces):
            self.window.blit(s, (rect.x + padding, rect.y + padding + j * line_height))

    def _process_events(self, env: WargameEnv) -> None:
        """Process pygame events for pause (Space), quit (Esc), and click-to-pin tooltip."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.should_quit = True
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
        self, canvas: pygame.Surface, deployment_zone: np.ndarray
    ) -> None:
        """Draw deployment zone on the canvas."""

        x = float(deployment_zone[0] * self.pix_square_size)
        y = float(deployment_zone[1] * self.pix_square_size)
        width = float((deployment_zone[2] - deployment_zone[0]) * self.pix_square_size)
        height = float((deployment_zone[3] - deployment_zone[1]) * self.pix_square_size)

        pygame.draw.rect(
            canvas,
            (200, 200, 200),
            pygame.Rect(x, y, width, height),
        )

    def _draw_deployment_zone_text(
        self, canvas: pygame.Surface, deployment_zone: np.ndarray
    ) -> None:
        """
        Draw the text "Deployment Zone" in the deployment zone.

        """
        font = pygame.font.Font(None, 48)
        text = font.render("Deployment Zone", True, (240, 240, 240))
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

    def _draw_target(
        self, canvas: pygame.Surface, objectives: list[WargameObjective]
    ) -> None:
        """Draw objectives on the canvas."""
        for objective in objectives:
            pygame.draw.circle(
                canvas,
                (255, 100, 100),
                (
                    float(objective.location[0] + 0.5) * self.pix_square_size,
                    float(objective.location[1] + 0.5) * self.pix_square_size,
                ),
                float(objective.radius_size * self.pix_square_size),
            )

            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    float(objective.location[0] * self.pix_square_size),
                    float(objective.location[1] * self.pix_square_size),
                    float(self.pix_square_size),
                    float(self.pix_square_size),
                ),
            )

    def _draw_agent(
        self, canvas: pygame.Surface, wargame_models: list[WargameModel]
    ) -> None:
        """Draw wargame models (agents) on the canvas."""
        for model in wargame_models:
            pygame.draw.circle(
                canvas,
                (0, 0, 255),
                (
                    float(model.location[0] + 0.5) * self.pix_square_size,
                    float(model.location[1] + 0.5) * self.pix_square_size,
                ),
                self.pix_square_size / 3,
            )

    def _draw_gridlines(self, canvas: pygame.Surface, size: int) -> None:
        """Draw gridlines on the canvas."""
        for x in range(size + 1):
            # Draw horizontal lines
            pygame.draw.line(
                canvas,
                0,
                (0, self.pix_square_size * x),
                (self.window_size, self.pix_square_size * x),
                width=3,
            )
            # Draw vertical lines
            pygame.draw.line(
                canvas,
                0,
                (self.pix_square_size * x, 0),
                (self.pix_square_size * x, self.window_size),
                width=3,
            )

    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
