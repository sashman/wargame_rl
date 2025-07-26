from typing import NamedTuple, TypedDict

import numpy as np


class WargameEnvConfig(NamedTuple):
    """
    Configuration for the Wargame environment.
    """

    number_of_wargame_models: int = 20  # Number of wargame models in the environment
    size: int = 50  # Size of the square grid
    render_mode: str | None = "human"  # Rendering mode for the environment
    deployment_zone: tuple[int, int, int, int] = (
        0,
        0,
        50,
        50,
    )  # Deployment zone coordinates


class WargameModelObservation(TypedDict):
    """
    Observation structure for a Wargame model.
    """

    location: np.ndarray  # Location of the wargame model in the grid


class WargameEnvObjectiveObservation(TypedDict):
    """
    Observation structure for a Wargame objective.
    """

    location: np.ndarray  # Location of the objective in the grid


class WargameEnvObservation(TypedDict):
    """
    Observation structure for the Wargame environment.
    """

    current_turn: int
    wargame_models: list[WargameModelObservation]
    objectives: list[WargameEnvObjectiveObservation]


class WargameEnvInfo(TypedDict):
    """
    Info structure for the Wargame environment.
    """

    current_turn: int
    wargame_models: list[WargameModelObservation]
    objectives: list[WargameEnvObjectiveObservation]
    deployment_zone: tuple[int, int, int, int]


class WargameEnvAction(TypedDict):
    """
    Action structure for the Wargame environment.

    List of lists, where each inner list contains:
    4 elements to represent the action for each wargame model: up (0), down (1), left (2), right (3).
    the outer list is for each wargame model, the length of the outer list is equal to the number of wargame models.
    """

    actions: list[list[int]]  # Actions for each wargame model
