from pydantic import BaseModel


class WargameEnvConfig(BaseModel):
    """
    Configuration for the Wargame environment.
    """

    number_of_wargame_models: int = 2  # Number of wargame models in the environment
    number_of_objectives: int = 2  # Number of objectives in the environment
    # Right now, this has to remain fixed. If changed, the model needs to be retrained.
    size: int = 50  # Size of the square grid
    render_mode: str | None = "human"  # Rendering mode for the environment
    deployment_zone: tuple[int, int, int, int] = (
        0,
        0,
        50,
        50,
    )  # Deployment zone coordinates
