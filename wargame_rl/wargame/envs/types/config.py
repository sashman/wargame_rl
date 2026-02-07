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
    group_max_distance: float = Field(
        gt=0,
        default=10.0,
        description="Max distance (L2) models in the same group may be from at least one other model in that group; violation yields group_violation_penalty.",
    )
    group_violation_penalty: float = Field(
        default=-10.0,
        description="Reward applied per model when it is farther than group_max_distance from every other model in its group.",
    )
