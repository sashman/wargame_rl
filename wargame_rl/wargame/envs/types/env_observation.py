from dataclasses import dataclass

from .model_observation import WargameModelObservation
from .objective_observation import WargameEnvObjectiveObservation


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
