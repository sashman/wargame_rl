from pydantic import BaseModel, Field, model_validator


class WargameEnvConfig(BaseModel):
    """
    Configuration for the Wargame environment.
    """

    config_name: str | None = Field(
        default=None, description="Name of the environment config"
    )
    number_of_wargame_models: int = 2  # Number of wargame models in the environment
    number_of_objectives: int = 2  # Number of objectives in the environment
    objective_radius_size: int = Field(
        gt=0, default=0, description="Radius of the objective in the environment"
    )
    board_width: int = Field(
        gt=0, default=50, description="Width of the grid (x dimension)"
    )
    board_height: int = Field(
        gt=0, default=50, description="Height of the grid (y dimension)"
    )
    render_mode: str | None = Field(
        default=None, description="Rendering mode for the environment"
    )
    deployment_zone: tuple[int, int, int, int] | None = Field(
        default=None,
        description="Deployment zone (x_min, y_min, x_max, y_max). If None, defaults to (0, 0, board_width//3, board_height).",
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

    @model_validator(mode="before")
    @classmethod
    def size_to_width_height(cls, data: object) -> object:
        """Backward compatibility: accept 'size' or 'width'/'height' in YAML/dict."""
        if not isinstance(data, dict):
            return data
        if "size" in data and "board_width" not in data and "board_height" not in data:
            s = data["size"]
            data = {**data, "board_width": s, "board_height": s}
        if "width" in data and "board_width" not in data:
            data = {**data, "board_width": data["width"]}
        if "height" in data and "board_height" not in data:
            data = {**data, "board_height": data["height"]}
        return data
