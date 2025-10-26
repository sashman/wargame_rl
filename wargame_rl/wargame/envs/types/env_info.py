from pydantic import BaseModel, ConfigDict, Field

from .model_observation import WargameModelObservation
from .objective_observation import WargameEnvObjectiveObservation


class WargameEnvInfo(BaseModel):
    """
    Info structure for the Wargame environment.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    current_turn: int = Field(description="The current turn number.")
    wargame_models: list[WargameModelObservation] = Field(
        description="The list of wargame models."
    )
    objectives: list[WargameEnvObjectiveObservation] = Field(
        description="The list of objectives."
    )
    deployment_zone: tuple[int, int, int, int] = Field(
        description="The deployment zone coordinates."
    )
