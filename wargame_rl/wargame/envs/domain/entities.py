"""Domain entities: WargameModel and WargameObjective (capture target)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from gymnasium import spaces

from wargame_rl.wargame.envs.domain.value_objects import (
    POSITION_DTYPE,
    Position,
    position,
    zero_position,
)
from wargame_rl.wargame.envs.types.geometry import Polygon

if TYPE_CHECKING:
    from wargame_rl.wargame.envs.reward.types.model_rewards import ModelRewards


class WargameModel:
    """
    A single model on the board. One model, not a whole group — see `group_id`.

    Args:
        location: Location of the model in the grid.
        stats: Statistics (e.g. wounds). Not used currently.
        distances_to_objectives: Distances to all objectives.
        group_id: The group this model belongs to — this project's name for the
            rules' *unit*. Models in a group stay close, spawn together, share a
            one-hot in the observation, and ignore each other (and their
            target's group) when tracing line of sight. Numbering is per army:
            player group 0 and opponent group 0 are different groups.
        base_radius: Physical radius of the model's base, in board units. 0.0
            makes it a dimensionless point.
        previous_closest_objective_distance: Used for reward shaping.
    """

    def __init__(
        self,
        location: Position,
        stats: dict[str, int],
        distances_to_objectives: np.ndarray,
        group_id: int,
        previous_closest_objective_distance: float | None = None,
        best_closest_objective_distance: float | None = None,
        base_radius: float = 0.0,
    ):
        self.location = location
        self.previous_location: Position | None = None
        self.stats = stats
        self.distances_to_objectives = distances_to_objectives
        self.group_id = group_id
        self.base_radius = base_radius

        self.previous_closest_objective_distance = previous_closest_objective_distance
        self.best_closest_objective_distance = best_closest_objective_distance
        self.model_rewards_history: list["ModelRewards"] = []
        self.advanced_this_turn: bool = False

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
        self.advanced_this_turn = False

    @property
    def is_alive(self) -> bool:
        """True while the model has wounds remaining."""
        return self.stats["current_wounds"] > 0

    @property
    def has_lost_wounds(self) -> bool:
        """True when the model is damaged but not destroyed.

        The allocation rule prefers a model that has *"already lost Wounds"*, so
        damage concentrates on one model rather than leaving a unit of survivors
        each one wound from death. Identically False at `max_wounds: 1`, where a
        model is either whole or gone.
        """
        return 0 < self.stats["current_wounds"] < self.stats["max_wounds"]

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
        # `board_width`, not `board_width - 1`. The `-1` was the last cell
        # *index*; on a continuous board the extent is the coordinate a model
        # standing on the far edge actually has.
        location_space = spaces.Box(
            low=zero_position(),
            high=position(board_width, board_height),
            shape=(2,),
            dtype=POSITION_DTYPE,
        )
        max_dx = float(max(board_width, board_height))
        distances_to_objectives_space = spaces.Box(
            low=-max_dx,
            high=max_dx,
            shape=(number_of_objectives, 2),
            dtype=POSITION_DTYPE,
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
    """Objective on the board: either a marker with a radius, or an area.

    An *area* objective is the rules' terrain objective — the ground itself is
    the prize, so control is standing inside the outline rather than within a
    distance of a point. It carries `radius_size = 0` and reports distance to its
    own edge, which is what lets every downstream `norms_offset <= obj_radii`
    test keep working with no branch: see `polygons_distance_to_points`.

    An area is not *placed*. Its outline is its position, and `location` is the
    centroid so anything steering toward an objective still has a point to aim
    at.
    """

    def __init__(
        self,
        location: Position,
        radius_size: float,
        area: Polygon | None = None,
    ):
        self.location = location
        self.radius_size = radius_size  # Radius of the objective in the environment
        self.area = area

    @property
    def is_area(self) -> bool:
        """True when the objective is a piece of ground rather than a marker."""
        return self.area is not None

    def set_area(self, area: Polygon) -> None:
        """Make this an area objective, moving its location to the centroid."""
        self.area = area
        self.location = position(*area.centroid)
        self.radius_size = 0.0

    def __repr__(self) -> str:
        if self.area is not None:
            return f"WargameObjective(area={self.area.bounds})"
        return f"WargameObjective(location={self.location}, radius_size={self.radius_size})"

    @staticmethod
    def to_space(board_width: int, board_height: int) -> spaces.Dict:
        """Gymnasium observation space for one objective (used by the env facade)."""
        return spaces.Dict(
            {
                "location": spaces.Box(
                    low=zero_position(),
                    high=position(board_width, board_height),
                    shape=(2,),
                    dtype=POSITION_DTYPE,
                )
            }
        )
