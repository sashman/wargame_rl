from __future__ import annotations

from enum import Enum

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from wargame_rl.wargame.envs.env_types import (
    WargameEnvAction,
    WargameEnvConfig,
    WargameEnvInfo,
    WargameEnvObjectiveObservation,
    WargameEnvObservation,
    WargameModelObservation,
)
from wargame_rl.wargame.envs.renders import renderer

# from wargame_rl.wargame.envs.renders.renderer import Renderer


class MovementPhaseActions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3
    none = 4


class WargameModel:
    def __init__(self, location, stats, distances_to_objectives):
        self.location = location  # Should be a numpy array of shape (2,)
        self.stats = (
            stats  # Should be a dictionary with keys 'max_wounds' and 'current_wounds'
        )
        self.distances_to_objectives = distances_to_objectives  # Should be a numpy array of shape (number_of_objectives,)

    def __repr__(self):
        return f"WargameModel(location={self.location}, distances_to_objectives={self.distances_to_objectives}, stats={self.stats})"


class WargameModelSpace:
    @staticmethod
    def to_space(size: int, number_of_objectives: int):
        location_space = spaces.Box(0, size - 1, shape=(2,), dtype=int)
        distances_to_objectives_space = spaces.Box(
            0, size - 1, shape=(number_of_objectives,), dtype=int
        )
        stats_space = spaces.Dict(
            {
                "max_wounds": spaces.Box(0, 100, shape=(1,), dtype=int),
                "current_wounds": spaces.Box(0, 100, shape=(1,), dtype=int),
            }
        )

        return spaces.Dict(
            {
                "location": location_space,
                "distances_to_objectives": distances_to_objectives_space,
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

    def __init__(
        self, config: WargameEnvConfig, renderer: renderer.Renderer | None = None
    ):
        self.size = config.size  # The size of the square grid
        self.window_size = 1024  # The size of the PyGame window
        self.config = config
        self.observation_space = spaces.Dict(
            {
                "current_turn": spaces.Discrete(1),
                "wargame_models": spaces.Tuple(
                    [
                        WargameModelSpace.to_space(
                            size=self.size,
                            number_of_objectives=config.number_of_objectives,
                        )
                        for _ in range(config.number_of_wargame_models)
                    ]
                ),
                "objectives": spaces.Sequence(
                    WargameObjectiveSpace.to_space(size=self.size)
                ),
            }
        )

        action_count = len(MovementPhaseActions)
        self.action_space = spaces.Tuple(
            [
                spaces.Discrete(action_count)
                for _ in range(config.number_of_wargame_models)
            ]
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
            MovementPhaseActions.none.value: np.array([0, 0]),
        }

        self.renderer = renderer

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
                distances_to_objectives=np.zeros(
                    config.number_of_objectives, dtype=int
                ),
            )
            for _ in range(config.number_of_wargame_models)
        ]
        self.previous_distance = [None] * config.number_of_wargame_models
        # List to hold objectives
        self.objectives = [
            WargameObjective(location=np.zeros(2, dtype=int))
            for _ in range(config.number_of_objectives)
        ]

        # Set the deployment zone for the agent, area left third of the grid
        self.deployment_zone = np.array([0, 0, self.size // 3, self.size], dtype=int)

    def _get_obs(self) -> WargameEnvObservation:
        """Get the observation for the current state of the environment."""
        # Get the locations of the wargame models and objectives

        for model in self.wargame_models:
            model.distances_to_objectives = np.array(
                [
                    np.linalg.norm(model.location - objective.location, ord=2)
                    for objective in self.objectives
                ],
                dtype=int,
            )

        wargame_models = [
            WargameModelObservation(
                location=model.location,
                distances_to_objectives=model.distances_to_objectives,
            )
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
            WargameModelObservation(
                location=model.location,
                distances_to_objectives=model.distances_to_objectives,
            )
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
        for i, model in enumerate(self.wargame_models):
            model.location = self.np_random.integers(
                self.deployment_zone[0], self.deployment_zone[2], size=2, dtype=int
            )
            model.stats["current_wounds"] = model.stats["max_wounds"]
            self.previous_distance[i] = None

        # For each objective, we will randomly choose a location outside the deployment zone
        for objective in self.objectives:
            objective.location = self.np_random.integers(
                self.deployment_zone[2], self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.renderer is not None:
            self.renderer.setup(self)
            self.renderer.render(self)

        return observation, info

    def _calculate_reward(self):
        """Calculate the reward based on the average negative normalized distance of all wargame models to the closest objectives."""
        total_distance = 0
        total_distance_improvement = 0
        for i, model in enumerate(self.wargame_models):
            closest_objective = min(
                self.objectives,
                key=lambda obj: np.linalg.norm(model.location - obj.location, ord=2),
            )
            distance = np.linalg.norm(
                model.location - closest_objective.location, ord=2
            )
            normalized_distance = distance / (np.sqrt(2) * self.size)
            total_distance += normalized_distance
            if distance == 0:
                model_reward = 1
            else:
                if self.previous_distance[i] is not None:
                    distance_improvement = distance - self.previous_distance[i]
                    if distance_improvement < 0:
                        model_reward = 0.05
                    elif distance_improvement > 0:
                        model_reward = -0.1
                    else:
                        model_reward = -0.05
                else:
                    model_reward = 0
            total_distance_improvement += model_reward
            self.previous_distance[i] = distance

        # average_distance = total_distance / len(self.wargame_models)
        # assert average_distance >= 0.0
        # assert average_distance <= 1.0
        # return -average_distance + total_distance_improvement / len(self.wargame_models)
        reward = total_distance_improvement / len(self.wargame_models)
        assert reward >= -1.0
        assert reward <= 1.0
        return reward

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
            for objective in self.objectives:
                if np.array_equal(model.location, objective.location):
                    terminated[i] = True
                    break

        is_terminated = all(
            terminated
        )  # If all models are terminated, the episode is done

        reward = self._calculate_reward()

        # reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        self.current_turn += 1  # Increment the current turn
        if self.current_turn >= self.max_turns:
            is_terminated = True

        return observation, reward, is_terminated, False, info

    def render(self):
        if self.renderer is not None:
            self.renderer.render(self)
