from enum import Enum

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from wargame_rl.wargame.envs.env_types import (
    WargameEnvAction,
    WargameEnvConfig,
    WargameEnvInfo,
    WargameEnvObjectiveObservation,
    WargameEnvObservation,
    WargameModelObservation,
)


class MovementPhaseActions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3


class WargameModel:
    def __init__(self, location, stats):
        self.location = location  # Should be a numpy array of shape (2,)
        self.stats = (
            stats  # Should be a dictionary with keys 'max_wounds' and 'current_wounds'
        )

    def __repr__(self):
        return f"WargameModel(location={self.location}, stats={self.stats})"


class WargameModelSpace:
    @staticmethod
    def to_space(size: int):
        location_space = spaces.Box(0, size - 1, shape=(2,), dtype=int)
        stats_space = spaces.Dict(
            {
                "max_wounds": spaces.Box(0, 100, shape=(1,), dtype=int),
                "current_wounds": spaces.Box(0, 100, shape=(1,), dtype=int),
            }
        )

        return spaces.Dict(
            {
                "location": location_space,
                "stats": stats_space,
            }
        )


class WargameObjective:
    def __init__(self, location):
        self.location = location  # Should be a numpy array of shape (2,)

    def __repr__(self):
        return f"WargameObjective(location={self.location})"


class WargameObjectiveSpace:
    @staticmethod
    def to_space(size: int):
        return spaces.Dict({"location": spaces.Box(0, size - 1, shape=(2,), dtype=int)})


class WargameEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(self, config: WargameEnvConfig):
        self.size = config.size  # The size of the square grid
        self.window_size = 1024  # The size of the PyGame window

        self.observation_space = spaces.Dict(
            {
                "current_turn": spaces.Discrete(1),
                "wargame_models": spaces.Tuple(
                    [
                        WargameModelSpace.to_space(size=self.size)
                        for _ in range(config.number_of_wargame_models)
                    ]
                ),
                "objectives": spaces.Sequence(
                    WargameObjectiveSpace.to_space(size=self.size)
                ),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right" for each wargame model
        self.action_space = spaces.Tuple(
            [spaces.Discrete(4) for _ in range(config.number_of_wargame_models)]
        )

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            MovementPhaseActions.right.value: np.array([1, 0]),
            MovementPhaseActions.up.value: np.array([0, 1]),
            MovementPhaseActions.left.value: np.array([-1, 0]),
            MovementPhaseActions.down.value: np.array([0, -1]),
        }

        assert (
            config.render_mode is None
            or config.render_mode in self.metadata["render_modes"]  # type: ignore
        )
        self.render_mode = config.render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        self.current_turn = 0  # Initialize the current turn to 0

        self.max_turns = self.size * 2  # Set the maximum number of turns

        # List to hold wargame models for each number_of_wargame_models
        self.wargame_models = [
            WargameModel(
                location=np.zeros(2, dtype=int),
                stats={"max_wounds": 100, "current_wounds": 100},
            )
            for _ in range(config.number_of_wargame_models)
        ]

        # List to hold objectives
        self.objectives = [
            WargameObjective(location=np.zeros(2, dtype=int))
            for _ in range(config.number_of_wargame_models)
        ]

        # Set the deployment zone for the agent, area left third of the grid
        self.deployment_zone = np.array([0, 0, self.size // 3, self.size], dtype=int)

    def _get_obs(self) -> WargameEnvObservation:
        """Get the observation for the current state of the environment."""
        # Get the locations of the wargame models and objectives
        wargame_models = [
            WargameModelObservation(location=model.location)
            for model in self.wargame_models
        ]
        objectives = [
            WargameEnvObjectiveObservation(location=objective.location)
            for objective in self.objectives
        ]
        # Create the observation dictionary
        return WargameEnvObservation(
            current_turn=self.current_turn,
            wargame_models=wargame_models,
            objectives=objectives,
        )

    def _get_info(self):
        # for each wargame model, we will return its location and stats

        wargame_models = [
            WargameModelObservation(location=model.location)
            for model in self.wargame_models
        ]
        objectives = [
            WargameEnvObjectiveObservation(location=objective.location)
            for objective in self.objectives
        ]
        # Create the observation dictionary
        return WargameEnvInfo(
            current_turn=self.current_turn,
            wargame_models=wargame_models,
            objectives=objectives,
            deployment_zone=self.deployment_zone.tolist(),
        )

    def reset(
        self, seed=None, options=None
    ) -> tuple[WargameEnvObservation, WargameEnvInfo]:
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset the current turn to 0
        self.current_turn = 0

        # For each wargame model, we will randomly choose a location within the deployment zone
        for model in self.wargame_models:
            model.location = self.np_random.integers(
                self.deployment_zone[0], self.deployment_zone[2], size=2, dtype=int
            )
            model.stats["current_wounds"] = model.stats["max_wounds"]

        # For each objective, we will randomly choose a location outside the deployment zone
        for objective in self.objectives:
            objective.location = self.np_random.integers(
                self.deployment_zone[2], self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _calculate_reward(self):
        """Calculate the reward based on the average negative normalized distance of all wargame models to the closest objectives."""
        total_distance = 0
        for model in self.wargame_models:
            closest_objective = min(
                self.objectives,
                key=lambda obj: np.linalg.norm(model.location - obj.location, ord=2),
            )
            distance = np.linalg.norm(
                model.location - closest_objective.location, ord=2
            )
            normalized_distance = distance / (np.sqrt(2) * self.size)
            total_distance += normalized_distance

        average_distance = total_distance / len(self.wargame_models)
        return -average_distance

    def step(
        self, action: WargameEnvAction
    ) -> tuple[WargameEnvObservation, float, bool, bool, WargameEnvInfo]:
        terminated = [False] * len(self.wargame_models)

        # for each element in the action tuple, we will move the corresponding wargame model
        for i, act in enumerate(action.actions):
            # Ensure the action is within the action space
            if not self.action_space[i].contains(act):
                raise ValueError(
                    f"Action {act} for wargame model {i} is out of bounds."
                )

            # Get the wargame model to move
            model = self.wargame_models[i]
            # Map the action (element of {0,1,2,3}) to the direction we walk in
            direction = self._action_to_direction[act]
            # We use `np.clip` to make sure we don't leave the grid
            model.location = np.clip(model.location + direction, 0, self.size - 1)

        # After moving all wargame models, we can check if any of them has reached its objective
        for i, model in enumerate(self.wargame_models):
            # Check if the model has reached its objective
            if np.array_equal(model.location, self.objectives[i].location):
                terminated[i] = True

        is_terminated = all(
            terminated
        )  # If all models are terminated, the episode is done

        reward = self._calculate_reward()

        # reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        self.current_turn += 1  # Increment the current turn
        if self.current_turn >= self.max_turns:
            is_terminated = True

        return observation, reward, is_terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        self._draw_target(canvas, pix_square_size)

        # Now we draw the agent
        self._draw_agent(canvas, pix_square_size)

        # Finally, add some gridlines
        self._draw_gridlines(canvas, pix_square_size)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def _draw_target(self, canvas, pix_square_size):
        """Draw objectives on the canvas."""
        for objective in self.objectives:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pix_square_size * objective.location,
                    (pix_square_size, pix_square_size),
                ),
            )

    def _draw_agent(self, canvas, pix_square_size):
        """Draw wargame models (agents) on the canvas."""
        for model in self.wargame_models:
            pygame.draw.circle(
                canvas,
                (0, 0, 255),
                (model.location + 0.5) * pix_square_size,
                pix_square_size / 3,
            )

    def _draw_gridlines(self, canvas, pix_square_size):
        """Draw gridlines on the canvas."""
        for x in range(self.size + 1):
            # Draw horizontal lines
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            # Draw vertical lines
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
