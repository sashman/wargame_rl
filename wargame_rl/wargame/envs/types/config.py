from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    """Per-model configuration (position, group, stats, etc.).

    When *x* and *y* are provided the model is placed at that exact cell;
    otherwise it is placed randomly in the deployment zone.
    """

    x: int | None = Field(
        default=None,
        ge=0,
        description="X coordinate on the board. If None, placed randomly.",
    )
    y: int | None = Field(
        default=None,
        ge=0,
        description="Y coordinate on the board. If None, placed randomly.",
    )
    group_id: int = Field(default=0, ge=0, description="Group this model belongs to")
    max_wounds: int = Field(default=100, gt=0)

    @model_validator(mode="after")
    def coords_both_or_neither(self) -> "ModelConfig":
        if (self.x is None) != (self.y is None):
            raise ValueError("x and y must both be set or both be None")
        return self


class ObjectiveConfig(BaseModel):
    """Per-objective configuration (position, radius, etc.).

    When *x* and *y* are provided the objective is placed at that exact cell;
    otherwise it is placed randomly outside the deployment zone.
    """

    x: int | None = Field(
        default=None,
        ge=0,
        description="X coordinate on the board. If None, placed randomly.",
    )
    y: int | None = Field(
        default=None,
        ge=0,
        description="Y coordinate on the board. If None, placed randomly.",
    )
    radius_size: int | None = Field(
        default=None,
        gt=0,
        description="Override the global objective_radius_size for this objective",
    )

    @model_validator(mode="after")
    def coords_both_or_neither(self) -> "ObjectiveConfig":
        if (self.x is None) != (self.y is None):
            raise ValueError("x and y must both be set or both be None")
        return self


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
        description="Player deployment zone (x_min, y_min, x_max, y_max). If None, defaults to (0, 0, board_width//3, board_height).",
    )
    opponent_deployment_zone: tuple[int, int, int, int] | None = Field(
        default=None,
        description="Opponent deployment zone (x_min, y_min, x_max, y_max). If None, defaults to (board_width*2//3, 0, board_width, board_height).",
    )
    models: list[ModelConfig] | None = Field(
        default=None,
        description="Per-model configuration (attributes, and optionally positions). Length must match number_of_wargame_models.",
    )
    objectives: list[ObjectiveConfig] | None = Field(
        default=None,
        description="Per-objective configuration (attributes, and optionally positions). Length must match number_of_objectives.",
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
    n_movement_angles: int = Field(
        gt=0,
        default=16,
        description="Number of angular bins for polar movement (e.g. 16 = 22.5° increments).",
    )
    n_speed_bins: int = Field(
        gt=0,
        default=6,
        description="Number of discrete speed levels from 1 to max_move_speed.",
    )
    max_move_speed: float = Field(
        gt=0,
        default=6.0,
        description="Maximum distance a model can move in a single step.",
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

    @property
    def has_fixed_model_positions(self) -> bool:
        """True when every model entry specifies x/y coordinates."""
        return self.models is not None and all(m.x is not None for m in self.models)

    @property
    def has_fixed_objective_positions(self) -> bool:
        """True when every objective entry specifies x/y coordinates."""
        return self.objectives is not None and all(
            o.x is not None for o in self.objectives
        )

    @model_validator(mode="after")
    def validate_entity_configs(self) -> "WargameEnvConfig":
        if self.models is not None:
            if len(self.models) != self.number_of_wargame_models:
                raise ValueError(
                    f"models has {len(self.models)} entries "
                    f"but number_of_wargame_models is {self.number_of_wargame_models}"
                )
            has_coords = [m.x is not None for m in self.models]
            if any(has_coords) and not all(has_coords):
                raise ValueError("Either all models must have x/y coordinates or none")
            for i, m in enumerate(self.models):
                if (
                    m.x is not None
                    and m.y is not None
                    and (m.x >= self.board_width or m.y >= self.board_height)
                ):
                    raise ValueError(
                        f"models[{i}] ({m.x}, {m.y}) is outside "
                        f"the board ({self.board_width}x{self.board_height})"
                    )
        if self.objectives is not None:
            if len(self.objectives) != self.number_of_objectives:
                raise ValueError(
                    f"objectives has {len(self.objectives)} entries "
                    f"but number_of_objectives is {self.number_of_objectives}"
                )
            has_coords = [o.x is not None for o in self.objectives]
            if any(has_coords) and not all(has_coords):
                raise ValueError(
                    "Either all objectives must have x/y coordinates or none"
                )
            for i, o in enumerate(self.objectives):
                if (
                    o.x is not None
                    and o.y is not None
                    and (o.x >= self.board_width or o.y >= self.board_height)
                ):
                    raise ValueError(
                        f"objectives[{i}] ({o.x}, {o.y}) is outside "
                        f"the board ({self.board_width}x{self.board_height})"
                    )
        return self
