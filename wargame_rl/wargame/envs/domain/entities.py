"""Domain entities: WargameModel (unit) and WargameObjective (capture target)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from gymnasium import spaces

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.reward.types.model_rewards import ModelRewards


class WargameModel:
    """
    Wargame model (unit) on the board.

    Args:
        location: Location of the model in the grid.
        stats: Statistics (e.g. wounds). Not used currently.
        distances_to_objectives: Distances to all objectives.
        group_id: Group ID; models in the same group are encouraged to stay close.
        previous_closest_objective_distance: Used for reward shaping.
    """

    def __init__(
        self,
        location: np.ndarray,
        stats: dict[str, int],
        distances_to_objectives: np.ndarray,
        group_id: int,
        previous_closest_objective_distance: float | None = None,
        best_closest_objective_distance: float | None = None,
    ):
        self.location = location
        self.previous_location: np.ndarray | None = None
        self.stats = stats
        self.distances_to_objectives = distances_to_objectives
        self.group_id = group_id

        self.previous_closest_objective_distance = previous_closest_objective_distance
        self.best_closest_objective_distance = best_closest_objective_distance
        self.model_rewards_history: list["ModelRewards"] = []

    def set_previous_closest_objective_distance(self, distance: float) -> None:
        self.previous_closest_objective_distance = distance

    def set_best_closest_objective_distance(self, distance: float) -> None:
        self.best_closest_objective_distance = distance

    def reset_for_episode(self) -> None:
        """Clear episode state before new placement (previous location, distances, rewards)."""
        self.previous_location = None
        self.previous_closest_objective_distance = None
        self.best_closest_objective_distance = None
        self.stats["current_wounds"] = self.stats["max_wounds"]
        self.model_rewards_history.clear()

    @property
    def is_alive(self) -> bool:
        """True while the model has wounds remaining."""
        return self.stats["current_wounds"] > 0

    def take_damage(self, amount: int) -> None:
        """Reduce current wounds by amount, clamped to 0.

        Sole entry point for wound reduction across the codebase.
        """
        self.stats["current_wounds"] = max(0, self.stats["current_wounds"] - amount)

    def __repr__(self) -> str:
        return f"WargameModel(location={self.location}, distances_to_objectives={self.distances_to_objectives}, group_id={self.group_id})"

    @staticmethod
    def to_space(
        board_width: int,
        board_height: int,
        number_of_objectives: int,
    ) -> spaces.Dict:
        """Gymnasium observation space for one model (used by the env facade)."""
        location_space = spaces.Box(
            low=np.array([0, 0], dtype=np.int32),
            high=np.array([board_width - 1, board_height - 1], dtype=np.int32),
            shape=(2,),
            dtype=np.int32,
        )
        max_dx = max(board_width, board_height) - 1
        distances_to_objectives_space = spaces.Box(
            low=-max_dx,
            high=max_dx,
            shape=(number_of_objectives, 2),
            dtype=np.int32,
        )
        stats_space = spaces.Dict(
            {
                "max_wounds": spaces.Box(0, 100, shape=(1,), dtype=np.int32),
                "current_wounds": spaces.Box(0, 100, shape=(1,), dtype=np.int32),
            }
        )
        alive_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            shape=(1,),
            dtype=np.float32,
        )

        group_id_space: spaces.Discrete = spaces.Discrete(1)

        return spaces.Dict(
            {
                "location": location_space,
                "distances_to_objectives": distances_to_objectives_space,
                "stats": stats_space,
                "alive": alive_space,
                "group_id": group_id_space,
            }
        )


def alive_mask_for(models: list[WargameModel]) -> np.ndarray:
    """Boolean array where True = model is alive. Used by distance cache, action masking, and reward."""
    return np.array([m.is_alive for m in models], dtype=bool)


class WargameObjective:
    """Objective (capture target) on the board."""

    def __init__(self, location: np.ndarray, radius_size: int):
        self.location = location  # numpy array of shape (2,)
        self.radius_size = radius_size  # Radius of the objective in the environment

    def __repr__(self) -> str:
        return f"WargameObjective(location={self.location}, radius_size={self.radius_size})"

    @staticmethod
    def to_space(board_width: int, board_height: int) -> spaces.Dict:
        """Gymnasium observation space for one objective (used by the env facade)."""
        return spaces.Dict(
            {
                "location": spaces.Box(
                    low=np.array([0, 0], dtype=np.int32),
                    high=np.array([board_width - 1, board_height - 1], dtype=np.int32),
                    shape=(2,),
                    dtype=np.int32,
                )
            }
        )
