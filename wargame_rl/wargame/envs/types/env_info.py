from dataclasses import dataclass

from .model_observation import WargameModelObservation
from .objective_observation import WargameEnvObjectiveObservation


@dataclass
class WargameEnvInfo:
    """
    Info structure for the Wargame environment.
    """

    current_turn: int
    wargame_models: list[WargameModelObservation]
    objectives: list[WargameEnvObjectiveObservation]
    deployment_zone: tuple[int, int, int, int]
