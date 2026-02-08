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
    group_cohesion_enabled: bool = Field(
        default=True,
        description="When True, models that break group cohesion receive group_violation_penalty; when False, no group cohesion reward is applied.",
    )
    group_max_distance: float = Field(
        gt=0,
        default=10.0,
        description="Max distance (L2) models in the same group may be from at least one other model in that group; violation yields group_violation_penalty (if group_cohesion_enabled).",
    )
    group_violation_penalty: float = Field(
        default=-10.0,
        description="Reward applied per model when it is farther than group_max_distance from every other model in its group.",
    )
    max_groups: int = Field(
        gt=0,
        default=100,
        description="Maximum number of groups in the game; group_id is one-hot encoded over this size for neural network input.",
    )
