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
        self._process_events()
        if self.should_quit:
            raise QuitRequested()
        self._render_frame(env)
        while self.paused:
            self._process_events()
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
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to
        # keep the framerate stable.
        self.clock.tick(metadata["render_fps"])

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
        turn_surface = font.render(turn_text, True, text_color)
        steps_surface = font.render(steps_text, True, text_color)
        reward_surface = font.render(reward_text, True, text_color)
        center_y = panel_y + self.PANEL_HEIGHT // 2
        turn_rect = turn_surface.get_rect(center=(self.window_size // 6, center_y))
        steps_rect = steps_surface.get_rect(center=(self.window_size // 2, center_y))
        reward_rect = reward_surface.get_rect(
            center=(5 * self.window_size // 6, center_y)
        )
        self.window.blit(turn_surface, turn_rect)
        self.window.blit(steps_surface, steps_rect)
        self.window.blit(reward_surface, reward_rect)

    def _process_events(self) -> None:
        """Process pygame events for pause (Space) and quit (Esc)."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.should_quit = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_ESCAPE:
                    self.should_quit = True

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
