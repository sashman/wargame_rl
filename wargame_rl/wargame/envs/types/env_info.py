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
    opponent_models: list[WargameModelObservation] = Field(
        default_factory=list,
        description="The list of opponent models.",
    )
    deployment_zone: tuple[int, int, int, int] = Field(
        description="The player deployment zone coordinates."
    )
    opponent_deployment_zone: tuple[int, int, int, int] = Field(
        description="The opponent deployment zone coordinates."
    )
    player_vp: int = Field(
        default=0, description="Cumulative victory points for the player."
    )
    opponent_vp: int = Field(
        default=0, description="Cumulative victory points for the opponent."
    )
    player_vp_delta: int = Field(
        default=0, description="VP added for the player during this step."
    )
    opponent_vp_delta: int = Field(
        default=0, description="VP added for the opponent during this step."
    )
