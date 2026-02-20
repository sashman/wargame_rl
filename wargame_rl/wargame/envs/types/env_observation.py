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
    board_width: int = 50
    board_height: int = 50

    @property
    def size_wargame_models(self) -> list[int]:
        return [model.size for model in self.wargame_models]

    @property
    def size_objectives(self) -> list[int]:
        return [objective.size for objective in self.objectives]

    @property
    def size_game_observation(self) -> int:
        return 1

    @property
    def size(self) -> int:
        total_size = (
            sum(self.size_wargame_models)
            + sum(self.size_objectives)
            + self.size_game_observation
        )
        return total_size

    @property
    def n_wargame_models(self) -> int:
        return len(self.wargame_models)

    @property
    def n_objectives(self) -> int:
        return len(self.objectives)
