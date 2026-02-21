from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from wargame_rl.wargame.envs.env_components import (
    ActionHandler,
    DistanceCache,
    build_info,
    build_observation,
    compute_distances,
    fixed_objective_placement,
    fixed_wargame_model_placement,
    get_termination,
    objective_placement,
    update_distances_to_objectives,
    wargame_model_placement,
)
from wargame_rl.wargame.envs.renders import renderer
from wargame_rl.wargame.envs.reward.reward import Reward
from wargame_rl.wargame.envs.types import (
    WargameEnvAction,
    WargameEnvConfig,
    WargameEnvInfo,
    WargameEnvObservation,
)
from wargame_rl.wargame.envs.wargame_model import WargameModel
from wargame_rl.wargame.envs.wargame_objective import WargameObjective

# Re-export for backward compatibility (tests, dqn import from here)
__all__ = ["WargameEnv", "WargameObjective"]


class WargameEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(
        self, config: WargameEnvConfig, renderer: renderer.Renderer | None = None
    ):
        self.board_width = config.board_width
        self.board_height = config.board_height
        self.window_size = 1024  # The size of the PyGame window
        self.config = config
        self.observation_space = spaces.Dict(
            {
                "current_turn": spaces.Discrete(1),
                "wargame_models": spaces.Tuple(
                    [
                        WargameModel.to_space(
                            board_width=self.board_width,
                            board_height=self.board_height,
                            number_of_objectives=config.number_of_objectives * 2,
                        )
                        for _ in range(config.number_of_wargame_models)
                    ]
                ),
                "objectives": spaces.Sequence(
                    WargameObjective.to_space(
                        board_width=self.board_width, board_height=self.board_height
                    )
                ),
            }
        )

        self._action_handler = ActionHandler(config)
        self.action_space = self._action_handler.action_space

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

        self.max_turns = (
            self.board_width + self.board_height
        )  # Set the maximum number of turns

        self.wargame_models = self.create_wargame_models(config)
        self.objectives = self.create_objectives(config)

        # Set the deployment zone for the agent, area left third of the grid
        if config.deployment_zone is not None:
            self.deployment_zone = np.array(config.deployment_zone, dtype=int)
        else:
            self.deployment_zone = np.array(
                [0, 0, self.board_width // 3, self.board_height], dtype=int
            )

        # Last reward from step(); None until first step after reset
        self.last_reward: float | None = None

    @staticmethod
    def create_wargame_models(config: WargameEnvConfig) -> list[WargameModel]:
        """Build the list of wargame models from config.

        When ``models`` are provided, group_id and max_wounds are taken from
        each entry.  Otherwise the models are split into groups automatically.
        """
        result: list[WargameModel] = []
        increment = max(1, config.number_of_wargame_models // config.max_groups)

        for i in range(config.number_of_wargame_models):
            if config.models is not None:
                mc = config.models[i]
                group_id = mc.group_id
                max_wounds = mc.max_wounds
            else:
                group_id = i // increment
                max_wounds = 100

            result.append(
                WargameModel(
                    location=np.zeros(2, dtype=int),
                    stats={"max_wounds": max_wounds, "current_wounds": max_wounds},
                    group_id=group_id,
                    distances_to_objectives=np.zeros(
                        [config.number_of_objectives, 2], dtype=int
                    ),
                )
            )
        return result

    @staticmethod
    def create_objectives(config: WargameEnvConfig) -> list[WargameObjective]:
        """Build the list of objectives from config.

        When ``objectives`` are provided, per-objective radius_size overrides
        the global value if set.
        """
        result: list[WargameObjective] = []
        for i in range(config.number_of_objectives):
            if (
                config.objectives is not None
                and config.objectives[i].radius_size is not None
            ):
                radius = config.objectives[i].radius_size
            else:
                radius = config.objective_radius_size

            result.append(
                WargameObjective(
                    location=np.zeros(2, dtype=int),
                    radius_size=radius,  # type: ignore[arg-type]
                )
            )
        return result

    def _get_obs(
        self, distance_cache: DistanceCache | None = None
    ) -> WargameEnvObservation:
        """Get the observation for the current state of the environment."""
        update_distances_to_objectives(
            self.wargame_models, self.objectives, distance_cache
        )
        return build_observation(
            self.current_turn,
            self.wargame_models,
            self.objectives,
            self.config.max_groups,
            self.board_width,
            self.board_height,
        )

    def _get_info(self) -> WargameEnvInfo:
        return build_info(
            self.current_turn,
            self.wargame_models,
            self.objectives,
            (
                int(self.deployment_zone[0]),
                int(self.deployment_zone[1]),
                int(self.deployment_zone[2]),
                int(self.deployment_zone[3]),
            ),
            self.config.max_groups,
        )

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WargameEnvObservation, dict[str, Any]]:
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.current_turn = 0
        self.last_reward = None

        if self.config.has_fixed_model_positions:
            fixed_wargame_model_placement(
                self.wargame_models,
                self.config.models,  # type: ignore[arg-type]
            )
        else:
            wargame_model_placement(
                self.wargame_models,
                self.deployment_zone,
                self.config.group_max_distance,
                self.np_random,
            )

        if self.config.has_fixed_objective_positions:
            fixed_objective_placement(
                self.objectives,
                self.config.objectives,  # type: ignore[arg-type]
            )
        else:
            objective_placement(
                self.objectives,
                self.deployment_zone,
                self.board_width,
                self.board_height,
                self.np_random,
            )

        cache = compute_distances(self.wargame_models, self.objectives)
        observation = self._get_obs(cache)
        info: WargameEnvInfo = self._get_info()

        if self.renderer is not None:
            self.renderer.setup(self)
            self.renderer.render(self)

        return observation, info.model_dump()

    def step(
        self, action: WargameEnvAction
    ) -> tuple[WargameEnvObservation, float, bool, bool, dict[str, Any]]:
        self._action_handler.apply(
            action,
            self.wargame_models,
            self.board_width,
            self.board_height,
            self._action_handler.action_space,
        )

        self.current_turn += 1

        cache = compute_distances(
            self.wargame_models,
            self.objectives,
            compute_model_model=self.config.group_cohesion_enabled,
        )

        is_terminated = get_termination(self.current_turn, self.max_turns, cache)

        reward = Reward().calculate_reward(self, cache)

        observation = self._get_obs(cache)
        info = self._get_info()

        self.last_reward = reward
        return observation, reward, is_terminated, False, info.model_dump()

    def render(self) -> None:
        if self.renderer is not None:
            self.renderer.render(self)

        return None
