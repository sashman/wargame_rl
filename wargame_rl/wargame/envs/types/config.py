from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, Field, field_validator, model_validator

from wargame_rl.wargame.envs.reward.phase import (
    RewardCalculatorConfig,
    RewardPhaseConfig,
    SuccessCriteriaConfig,
)
from wargame_rl.wargame.envs.types.game_timing import NON_MOVEMENT_PHASES, BattlePhase


class _HasCoords(Protocol):
    x: int | None
    y: int | None


_CoordsT = TypeVar("_CoordsT", bound=_HasCoords)


def _validate_coords_both_or_neither(x: int | None, y: int | None) -> None:
    """Raise if exactly one of x, y is None."""
    if (x is None) != (y is None):
        raise ValueError("x and y must both be set or both be None")


def _validate_entity_configs(
    count: int,
    configs: list[_CoordsT] | None,
    board_width: int,
    board_height: int,
    entity_name: str,
) -> None:
    """Validate entity list length, all-or-none coords, and in-bounds for fixed positions."""
    if configs is None:
        return
    if len(configs) != count:
        raise ValueError(
            f"{entity_name} has {len(configs)} entries but expected {count}"
        )
    has_coords = [c.x is not None for c in configs]
    if any(has_coords) and not all(has_coords):
        raise ValueError(f"Either all {entity_name} must have x/y coordinates or none")
    for i, c in enumerate(configs):
        if (
            c.x is not None
            and c.y is not None
            and (c.x >= board_width or c.y >= board_height)
        ):
            raise ValueError(
                f"{entity_name}[{i}] ({c.x}, {c.y}) is outside "
                f"the board ({board_width}x{board_height})"
            )


def _normalise_rect(r: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = r
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


# Rejection sampling for random terrain slows sharply as the board fills, so
# layouts are rejected at config load well before that point.
_MAX_TERRAIN_PACKING_FRACTION = 0.5


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 <= bx1 and bx0 <= ax1 and ay0 <= by1 and by0 <= ay1


class TurnOrder(str, Enum):
    """Who moves first each turn."""

    player = "player"
    opponent = "opponent"
    random = "random"


class OpponentPolicyConfig(BaseModel):
    """Configuration for the opponent policy engine."""

    type: str = Field(
        description="Policy engine identifier, e.g. 'random', 'scripted_advance_to_objective'."
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Policy-specific parameters forwarded to the policy constructor.",
    )


class MissionConfig(BaseModel):
    """Configuration for the mission (victory point scoring rules)."""

    type: str = Field(
        default="default",
        description="Mission type identifier; selects the VP calculator (e.g. 'default', 'none').",
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Mission-specific parameters (e.g. vp_per_objective, cap_per_turn, min_round).",
    )


class WeaponProfile(BaseModel):
    """Weapon stat block with range and resolution stats."""

    range: int = Field(gt=0, description="Maximum range in grid cells")
    attacks: int = Field(
        default=2, gt=0, description="Number of hit rolls per shooting action"
    )
    ballistic_skill: int = Field(
        default=3, ge=2, le=6, description="D6 roll needed to hit (e.g. 3 means 3+)"
    )
    strength: int = Field(
        default=4, gt=0, description="For wound roll comparison vs target toughness"
    )
    ap: int = Field(
        default=1,
        ge=0,
        description="Armour penetration (worsens target save by this amount)",
    )
    damage: int = Field(default=1, gt=0, description="Wounds inflicted per failed save")


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
    max_wounds: int = Field(default=1, gt=0)
    toughness: int = Field(default=3, gt=0, description="Wound roll comparison stat")
    save: int = Field(
        default=4,
        ge=2,
        le=7,
        description="Base armour save (e.g. 4 means 4+, 7 means no armour)",
    )
    weapons: list[WeaponProfile] = Field(
        default_factory=list,
        description="Weapon profiles. Empty = cannot shoot.",
    )

    @model_validator(mode="after")
    def coords_both_or_neither(self) -> "ModelConfig":
        _validate_coords_both_or_neither(self.x, self.y)
        return self


class TerrainPieceConfig(BaseModel):
    """Configuration for a single terrain piece (axis-aligned rectangle)."""

    footprint: tuple[int, int, int, int] = Field(
        description="Bounding rectangle (x0, y0, x1, y1) in grid cells."
    )


class RandomTerrainConfig(BaseModel):
    """Regenerate terrain footprints randomly at the start of every episode.

    The piece *count* is fixed while size and position vary. This is a hard
    constraint, not a simplification: `observations_to_tensor_batch` stacks the
    terrain arrays of a whole batch with `np.stack`, and `MLPNetwork` flattens
    terrain into a fixed-width input, so a batch containing episodes with
    different piece counts cannot be collated.

    Randomising terrain is what makes a cover result falsifiable. With a fixed
    layout a policy can memorise a handful of rectangles; with a fresh layout
    every episode it has to read the terrain tokens in the observation.
    """

    count: int = Field(
        gt=0,
        default=7,
        description="Number of terrain pieces. Constant across episodes.",
    )
    min_size: int = Field(
        gt=0, default=5, description="Minimum footprint side length in cells."
    )
    max_size: int = Field(
        gt=0, default=7, description="Maximum footprint side length in cells."
    )
    mirror: bool = Field(
        default=True,
        description="Mirror the layout across the vertical centre line. Deployment "
        "zones are fixed to the left and right of the board, so an asymmetric "
        "random layout would systematically favour one side.",
    )
    edge_margin: int = Field(
        ge=0, default=2, description="Keep footprints this far from the board edge."
    )
    min_gap: int = Field(
        ge=0,
        default=1,
        description="Minimum clear cells between two footprints. 0 lets them touch.",
    )

    @model_validator(mode="after")
    def sizes_ordered(self) -> "RandomTerrainConfig":
        """Reject an inverted size range."""
        if self.min_size > self.max_size:
            raise ValueError(
                f"min_size ({self.min_size}) must not exceed max_size ({self.max_size})"
            )
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
        _validate_coords_both_or_neither(self.x, self.y)
        return self


def _default_reward_phases() -> list[RewardPhaseConfig]:
    """Single default phase: reach objectives (closest_objective only)."""
    return [
        RewardPhaseConfig(
            name="reach_objectives",
            reward_calculators=[
                RewardCalculatorConfig(type="closest_objective", weight=1.0),
            ],
            success_criteria=SuccessCriteriaConfig(type="all_at_objectives"),
        )
    ]


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
        gt=0, default=1, description="Radius of the objective in the environment"
    )
    board_width: int = Field(
        gt=0, default=50, description="Width of the grid (x dimension)"
    )
    board_height: int = Field(
        gt=0, default=50, description="Height of the grid (y dimension)"
    )
    blocking_mask: list[list[bool]] | None = Field(
        default=None,
        description=(
            "Optional LOS blocking grid: outer list is y (row 0..board_height-1), "
            "inner is x (column 0..board_width-1). Cells True block line-of-sight "
            "through interior path cells only. None = no terrain blocking."
        ),
    )
    terrain: list[TerrainPieceConfig] | None = Field(
        default=None,
        description="Terrain pieces that block LOS. Each piece is an axis-aligned "
        "rectangle defined by a (x0, y0, x1, y1) footprint. None = no terrain.",
    )
    random_terrain: RandomTerrainConfig | None = Field(
        default=None,
        description="Regenerate terrain randomly each episode instead of using a "
        "fixed `terrain` list. Mutually exclusive with `terrain`. None = fixed.",
    )
    track_exposure: bool = Field(
        default=False,
        description="Accumulate line-of-sight exposure and terrain-proximity "
        "statistics during shooting phases. Measurement only — it does not affect "
        "the game, but it costs an extra shooting-mask build per shooting phase.",
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
    objective_min_separation: int | None = Field(
        default=None,
        ge=0,
        description="Minimum distance between two objective centres when placed "
        "randomly. None (default) places each independently, which lets discs "
        "overlap — measured at 25% of episodes on a 60x44 board with 3 objectives "
        "of radius 3. Set to 2 x objective_radius_size for disjoint discs.",
    )
    objective_terrain_clearance: int | None = Field(
        default=None,
        ge=0,
        description="Minimum distance from an objective centre to any terrain "
        "footprint. None (default) allows objectives inside ruins. Set it to keep "
        "the contested ground in the open, so terrain is cover on the approach "
        "rather than something standing on the prize.",
    )
    group_max_distance: float = Field(
        gt=0,
        default=10.0,
        description="Max distance (L2) for group-aware placement on reset: models in the same group spawn within this distance. Reward phases use their own group_cohesion params.",
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
    reward_phases: list[RewardPhaseConfig] = Field(
        default_factory=_default_reward_phases,
        min_length=1,
        description="Ordered reward phases for curriculum learning. "
        "Each phase defines reward calculators and success criteria for advancement.",
    )
    terminal_success_bonus: float = Field(
        default=0.0,
        description="Deprecated: use terminal_success_bonus on RewardPhaseConfig instead. "
        "Applied only to phases that do not define their own value.",
    )
    terminal_vp_bonus: float = Field(
        default=0.0,
        description="Deprecated: use terminal_vp_bonus on RewardPhaseConfig instead. "
        "Applied only to phases that do not define their own value.",
    )

    skip_phases: list[BattlePhase] = Field(
        default_factory=lambda: list(NON_MOVEMENT_PHASES),
        description="Battle phases to auto-advance through (the agent never steps "
        "on these). Defaults to all non-movement phases. Set to [] to "
        "step through every phase.",
    )

    terminate_on_player_elimination: bool = Field(
        default=False,
        description="If True, episode ends when all player models are eliminated. "
        "If False (default, matching tabletop rules), the opponent continues "
        "playing and scoring VP after wiping the player.",
    )

    number_of_battle_rounds: int = Field(
        default=100,
        gt=0,
        description=(
            "Number of battle rounds per game and the sole control over episode "
            "length. Default 100 gives training-length episodes (the tabletop "
            "standard is 5; set it explicitly for rules-faithful games). Episode "
            "length = this value x active phases per round (1 step per round with "
            "the default skip_phases)."
        ),
    )

    # --- Opponent configuration ---
    number_of_opponent_models: int = Field(
        default=0,
        ge=0,
        description="Number of opponent models. 0 means no opponents (backward-compatible).",
    )
    opponent_models: list[ModelConfig] | None = Field(
        default=None,
        description="Per-opponent-model configuration (reuses ModelConfig). "
        "Length must match number_of_opponent_models.",
    )
    turn_order: TurnOrder = Field(
        default=TurnOrder.player,
        description="Who moves first: 'player', 'opponent', or 'random' (coin-flip each step).",
    )
    opponent_policy: OpponentPolicyConfig | None = Field(
        default=None,
        description="Opponent policy engine config. Required when number_of_opponent_models > 0.",
    )
    mission: MissionConfig = Field(
        default_factory=MissionConfig,
        description="Mission config: selects VP calculator and params (vp_per_objective, cap_per_turn, min_round).",
    )

    @field_validator("blocking_mask", mode="before")
    @classmethod
    def normalize_blocking_mask(cls, value: object) -> object:
        """Allow YAML 0/1 integers as well as booleans."""
        if value is None:
            return None
        if not isinstance(value, list):
            raise TypeError("blocking_mask must be a list of rows or None")
        rows: list[list[bool]] = []
        for i, row in enumerate(value):
            if not isinstance(row, list):
                raise TypeError(f"blocking_mask row {i} must be a list")
            out_row: list[bool] = []
            for j, cell in enumerate(row):
                if isinstance(cell, bool):
                    out_row.append(cell)
                elif cell in (0, 1):
                    out_row.append(cell == 1)
                else:
                    raise ValueError(
                        f"blocking_mask cell [{i}][{j}] must be bool or 0/1, got {cell!r}"
                    )
            rows.append(out_row)
        return rows

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

    @property
    def has_fixed_opponent_positions(self) -> bool:
        """True when every opponent model entry specifies x/y coordinates."""
        return self.opponent_models is not None and all(
            m.x is not None for m in self.opponent_models
        )

    @model_validator(mode="after")
    def apply_legacy_terminal_bonus_defaults(self) -> "WargameEnvConfig":
        """Backfill per-phase terminal bonuses from deprecated env-level fields."""
        if not self.reward_phases:
            return self

        updated_phases: list[RewardPhaseConfig] = []
        for phase in self.reward_phases:
            updates: dict[str, float] = {}

            phase_has_success_bonus = "terminal_success_bonus" in phase.model_fields_set
            if not phase_has_success_bonus and self.terminal_success_bonus != 0.0:
                updates["terminal_success_bonus"] = self.terminal_success_bonus

            phase_has_vp_bonus = "terminal_vp_bonus" in phase.model_fields_set
            if not phase_has_vp_bonus and self.terminal_vp_bonus != 0.0:
                updates["terminal_vp_bonus"] = self.terminal_vp_bonus

            updated_phases.append(
                phase if not updates else phase.model_copy(update=updates)
            )

        self.reward_phases = updated_phases
        return self

    @model_validator(mode="after")
    def validate_blocking_mask_shape(self) -> "WargameEnvConfig":
        if self.blocking_mask is None:
            return self
        if len(self.blocking_mask) != self.board_height:
            raise ValueError(
                "blocking_mask must have board_height rows "
                f"({self.board_height}), got {len(self.blocking_mask)}"
            )
        for yi, row in enumerate(self.blocking_mask):
            if len(row) != self.board_width:
                raise ValueError(
                    "blocking_mask row "
                    f"{yi} must have length board_width ({self.board_width}), "
                    f"got {len(row)}"
                )
        return self

    @model_validator(mode="after")
    def validate_terrain(self) -> "WargameEnvConfig":
        """Validate terrain footprints are in-bounds and non-overlapping."""
        if self.terrain is None:
            return self
        normalised: list[tuple[int, int, int, int]] = [
            _normalise_rect(t.footprint) for t in self.terrain
        ]
        for i, (x0, y0, x1, y1) in enumerate(normalised):
            if x0 < 0 or y0 < 0 or x1 >= self.board_width or y1 >= self.board_height:
                raise ValueError(
                    f"terrain[{i}] {self.terrain[i].footprint} is outside "
                    f"the board ({self.board_width}x{self.board_height})"
                )
        for i in range(len(normalised)):
            for j in range(i + 1, len(normalised)):
                if _rects_overlap(normalised[i], normalised[j]):
                    raise ValueError(f"terrain[{i}] overlaps terrain[{j}]")
        return self

    @model_validator(mode="after")
    def validate_random_terrain(self) -> "WargameEnvConfig":
        """Reject a random-terrain spec that cannot be satisfied.

        Generation is rejection sampling, so an over-packed board would fail
        deep inside a training run rather than at load. Everything checkable
        from the config is checked here instead.
        """
        spec = self.random_terrain
        if spec is None:
            return self
        if self.terrain is not None:
            raise ValueError(
                "random_terrain and terrain are mutually exclusive: "
                "terrain is either fixed or regenerated each episode, not both"
            )

        usable_width = self.board_width - 2 * spec.edge_margin
        usable_height = self.board_height - 2 * spec.edge_margin
        if spec.max_size > usable_width or spec.max_size > usable_height:
            raise ValueError(
                f"random_terrain.max_size ({spec.max_size}) does not fit inside the "
                f"board minus edge_margin ({usable_width}x{usable_height})"
            )
        if spec.mirror and 2 * (spec.max_size + spec.min_gap) > usable_width:
            raise ValueError(
                "random_terrain.mirror needs room for a footprint and its mirror "
                f"image: 2 x (max_size + min_gap) = "
                f"{2 * (spec.max_size + spec.min_gap)} exceeds the usable width "
                f"({usable_width})"
            )

        # Rejection sampling degrades badly well before the board is full, so
        # the ceiling is a packing fraction rather than a strict area fit.
        #
        # Bounds the *expected* footprint, not the worst case. Sides are drawn
        # independently and uniformly from [min_size, max_size], so an all-
        # max_size layout is vanishingly unlikely, and bounding it rejects
        # perfectly generatable specs -- notably any spec wide enough to produce
        # walls, whose whole point is a large max_size next to a small min_size.
        # `_MAX_LAYOUT_ATTEMPTS` in terrain_placement.py is the real backstop:
        # it raises with a clear message if a draw genuinely cannot be placed.
        mean_size = (spec.min_size + spec.max_size) / 2
        required = spec.count * (mean_size + spec.min_gap) ** 2
        usable = usable_width * usable_height
        if required > _MAX_TERRAIN_PACKING_FRACTION * usable:
            raise ValueError(
                f"random_terrain packs too tightly: {spec.count} pieces averaging "
                f"{mean_size:g}x{mean_size:g} (plus {spec.min_gap} gap) need "
                f"{required:g} cells, more than "
                f"{_MAX_TERRAIN_PACKING_FRACTION:.0%} of the usable {usable}"
            )
        return self

    @model_validator(mode="after")
    def validate_entity_configs(self) -> "WargameEnvConfig":
        _validate_entity_configs(
            self.number_of_wargame_models,
            self.models,
            self.board_width,
            self.board_height,
            "models",
        )
        _validate_entity_configs(
            self.number_of_objectives,
            self.objectives,
            self.board_width,
            self.board_height,
            "objectives",
        )
        if self.number_of_opponent_models > 0 and self.opponent_policy is None:
            raise ValueError(
                "opponent_policy must be set when number_of_opponent_models > 0"
            )
        _validate_entity_configs(
            self.number_of_opponent_models,
            self.opponent_models,
            self.board_width,
            self.board_height,
            "opponent_models",
        )
        return self
