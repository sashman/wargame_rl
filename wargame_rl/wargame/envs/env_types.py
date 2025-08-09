from dataclasses import dataclass

import numpy as np


@dataclass
class WargameEnvConfig:
    """
    Configuration for the Wargame environment.
    """

    number_of_wargame_models: int = 1  # Number of wargame models in the environment
    # Right now, this has to remain fixed. If changed, the model needs to be retrained.
    size: int = 20  # Size of the square grid
    render_mode: str | None = "human"  # Rendering mode for the environment
    deployment_zone: tuple[int, int, int, int] = (
        0,
        0,
        50,
        50,
    )  # Deployment zone coordinates


@dataclass
class WargameModelObservation:
    """
    Observation structure for a Wargame model.
    """

    location: np.ndarray  # Location of the wargame model in the grid

    @property
    def size(self) -> int:
        return self.location.size


@dataclass
class WargameEnvObjectiveObservation:
    """
    Observation structure for a Wargame objective.
    """

    location: np.ndarray  # Location of the objective in the grid

    @property
    def size(self) -> int:
        return self.location.size


@dataclass
class WargameEnvObservation:
    """
    Observation structure for the Wargame environment.
    """

    current_turn: int
    wargame_models: list[WargameModelObservation]
    objectives: list[WargameEnvObjectiveObservation]

    @property
    def size(self) -> int:
        size_wargame_models = sum(model.size for model in self.wargame_models)
        size_objectives = sum(objective.size for objective in self.objectives)
        total_size = size_wargame_models + size_objectives + 1
        return total_size

    @property
    def n_wargame_models(self) -> int:
        return len(self.wargame_models)


@dataclass
class WargameEnvInfo:
    """
    Info structure for the Wargame environment.
    """

    current_turn: int
    wargame_models: list[WargameModelObservation]
    objectives: list[WargameEnvObjectiveObservation]
    deployment_zone: tuple[int, int, int, int]


@dataclass
class WargameEnvAction:
    """
    Action structure for the Wargame environment.

    List of ints, where each int contains the action for each wargame model: up (0), down (1), left (2), right (3).
    The length of the list is equal to the number of wargame models.
    """

    actions: list[int]  # Actions for each wargame model
