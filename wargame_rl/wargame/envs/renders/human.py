import pygame

from wargame_rl.wargame.envs.renders.renderer import Renderer
from wargame_rl.wargame.envs.wargame import WargameEnv


class HumanRender(Renderer):
    def __init__(self):
        self.window = None
        self.clock = None
        self.window_size = 1024
        self.canvas = None

    def setup(self, env: WargameEnv):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.canvas = pygame.Surface((self.window_size, self.window_size))
        self.canvas.fill((255, 255, 255))
        self.pix_square_size = (
            self.window_size / env.config.size
        )  # The size of a single grid square in pixels

    def render(self, env: WargameEnv):
        self._render_frame(env)

    def _render_frame(self, env: WargameEnv):
        env = env
        size = env.config.size
        objectives = env.objectives
        wargame_models = env.wargame_models
        metadata = env.metadata
        deployment_zone = env.deployment_zone

        # Clear the canvas
        self.canvas.fill((255, 255, 255))

        self._draw_deployment_zone(deployment_zone)

        self._draw_deployment_zone_text(deployment_zone)

        # We draw the target
        self._draw_target(objectives)

        # Now we draw the agent
        self._draw_agent(wargame_models)

        # Finally, add some gridlines
        self._draw_gridlines(size)

        # The following line copies our drawings from `canvas` to the visible window
        self.window.blit(self.canvas, self.canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to
        # keep the framerate stable.
        self.clock.tick(metadata["render_fps"])

    def _draw_deployment_zone(self, deployment_zone):
        """Draw deployment zone on the canvas."""

        deployment_zone = (
            float(deployment_zone[0] * self.pix_square_size),
            float(deployment_zone[1] * self.pix_square_size),
            float(deployment_zone[2] * self.pix_square_size),
            float(deployment_zone[3] * self.pix_square_size),
        )

        pygame.draw.rect(
            self.canvas,
            (200, 200, 200),
            pygame.Rect(deployment_zone),
        )

    def _draw_deployment_zone_text(self, deployment_zone):
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

        self.canvas.blit(
            text,
            (
                text_x * self.pix_square_size - text_width / 2,
                text_y * self.pix_square_size - text_height / 2,
            ),
        )

    def _draw_target(self, objectives):
        """Draw objectives on the canvas."""
        for objective in objectives:
            pygame.draw.rect(
                self.canvas,
                (255, 0, 0),
                pygame.Rect(
                    self.pix_square_size * objective.location,
                    (self.pix_square_size, self.pix_square_size),
                ),
            )

    def _draw_agent(self, wargame_models):
        """Draw wargame models (agents) on the canvas."""
        for model in wargame_models:
            pygame.draw.circle(
                self.canvas,
                (0, 0, 255),
                (model.location + 0.5) * self.pix_square_size,
                self.pix_square_size / 3,
            )

    def _draw_gridlines(self, size):
        """Draw gridlines on the canvas."""
        for x in range(size + 1):
            # Draw horizontal lines
            pygame.draw.line(
                self.canvas,
                0,
                (0, self.pix_square_size * x),
                (self.window_size, self.pix_square_size * x),
                width=3,
            )
            # Draw vertical lines
            pygame.draw.line(
                self.canvas,
                0,
                (self.pix_square_size * x, 0),
                (self.pix_square_size * x, self.window_size),
                width=3,
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
