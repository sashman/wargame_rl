from pydantic import BaseModel, Field


class WargameEnvConfig(BaseModel):
    """
    Configuration for the Wargame environment.
    """

    number_of_wargame_models: int = 2  # Number of wargame models in the environment
    number_of_objectives: int = 2  # Number of objectives in the environment
    objective_radius_size: int = Field(
        gt=0, default=0, description="Radius of the objective in the environment"
    )
    size: int = Field(gt=0, default=50, description="Size of the square grid")
    render_mode: str | None = Field(
        default=None, description="Rendering mode for the environment"
    )
    deployment_zone: tuple[int, int, int, int] = Field(
        default=(0, 0, 50, 50), description="Deployment zone coordinates"
    )
