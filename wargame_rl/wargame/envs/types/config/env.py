"""The environment configuration: the whole scenario in one model."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from wargame_rl.wargame.envs.reward.phase import (
    RewardCalculatorConfig,
    RewardPhaseConfig,
    SuccessCriteriaConfig,
)
from wargame_rl.wargame.envs.types.config._validation import (
    _MAX_TERRAIN_PACKING_FRACTION,
    _validate_entity_configs,
)
from wargame_rl.wargame.envs.types.config.battle import (
    MissionConfig,
    OpponentPolicyConfig,
    TurnOrder,
)
from wargame_rl.wargame.envs.types.config.entities import ModelConfig, ObjectiveConfig
from wargame_rl.wargame.envs.types.config.terrain import (
    RandomTerrainConfig,
    TerrainPieceConfig,
)
from wargame_rl.wargame.envs.types.game_timing import NON_MOVEMENT_PHASES, BattlePhase


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

    model_config = ConfigDict(extra="forbid")

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
    objectives_spread_on_terrain: bool = Field(
        default=False,
        description=(
            "With `objectives_on_terrain`, choose the eligible pieces whose "
            "minimum pairwise separation is largest instead of the ones nearest "
            "the board centre. Nearest-to-centre packs all three objectives into "
            "a ~16 inch circle on a 60x44 board, with 47% of pairs inside one "
            "weapon range, so there is no travel trade-off between them. "
            "Defaults False: turning it on changes the scenario, so every "
            "baseline measured without it must be re-measured."
        ),
    )
    start_on_objective_probability: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Probability that a training episode starts with one whole player "
            "group already standing on a random objective, instead of in the "
            "deployment zone. A start-state augmentation, not a rule: it "
            "teleports a squad, and it applies only when the caller asks for it "
            "via `reset(options={'augment_start': True})`, which the training "
            "rollout does and no evaluation path does. Aimed at a measured "
            "optimisation failure rather than a pricing one — putting a squad on "
            "the objective the trained agent abandons is worth +3.26 episode "
            "reward against a travel cost of ~0.27, and it still does not go. "
            "Defaults 0.0, which is an exact no-op: the augmentation draws "
            "nothing from the layout RNG unless it is both requested and "
            "positive. Note it draws whenever those hold, *including on the "
            "episodes where it does not fire — so at a probability below 1.0 "
            "the non-firing episodes are not stream-identical to a control "
            "run, and must not be treated as a matched within-run control."
        ),
    )
    observe_objective_control: bool = Field(
        default=False,
        description="Put per-objective control state (player count, opponent "
        "count, radius) on the objective token, widening it from 2 to 5. VP is "
        "scored on `player_count > opponent_count` per objective, but an "
        "objective otherwise reaches the network as nothing but a location, so "
        "the agent is asked to optimise a strict count comparison it cannot "
        "observe. Any reward keyed on those counts is likewise unattributable. "
        "Default False keeps the tensor byte-identical; turning it on changes "
        "the objective embedding shape, so existing checkpoints will fail to "
        "load — which is the intended loud failure.",
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
    objectives_on_terrain: bool = Field(
        default=False,
        description="Make each objective *be* a terrain piece: the pieces "
        "nearest the board centre, outside both deployment zones, become area "
        "objectives whose outline is the footprint. This is the rules' terrain "
        "objective — the ground itself is the prize — and it puts cover and the "
        "contested ground in the same place, which is the opposite of what "
        "`objective_terrain_clearance` arranges. Needs enough eligible pieces "
        "for the objective count, and fails loudly when there are not.",
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
        description="The scenario's coherency distance, in inches. Models of a group spawn within it, and `group_cohesion` fines a model past it unless that phase overrides `group_max_distance` explicitly. **One number for one concept**: these were independent, and every shipped config set placement to the 10.0 default while fining anything past 6.0, so 199 of 200 episodes started in violation. The rules' own figures are 2\" chaining and 9\" spread; adopting those is a scenario change and its own measured step.",
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
        description="Maximum distance a model can move in a single step, in inches.",
    )
    inches_per_unit: float = Field(
        gt=0,
        default=1.0,
        description=(
            "How many rules inches one board coordinate unit spans. Rules "
            "distances (move, weapon range, engagement) are authored in inches "
            "and divided by this to compare against coordinates; positions, "
            "board size and terrain footprints are already in units. At the "
            "default of 1.0 the two coincide and every conversion is the "
            "identity, which is why introducing the scale changed no result."
        ),
    )
    engagement_range: float = Field(
        gt=0,
        default=1.0,
        description=(
            "How close an enemy must be, in inches, for a model to count as "
            "engaged and be unable to shoot. Was a hard-coded constant. The "
            "rules say 2 (docs/rules/constants.yaml, engagement.horizontal_in); "
            "1 is kept as the default because every baseline and trained result "
            "in the repo was measured at 1, and raising it changes which shots "
            "are legal. See docs/rules/implementation-status.md."
        ),
    )
    base_radius: float = Field(
        ge=0,
        default=0.0,
        description=(
            "Radius of a model's base, in inches. Gives a model a physical "
            "extent: bases may not overlap at placement, objective range is "
            "measured from the base edge rather than its centre, and the "
            "renderer draws it at this size. 0.0 (the default) keeps models "
            "dimensionless points, which is what every result measured before "
            "continuous space assumed. The rules' infantry base is 32mm across "
            "-- a radius of about 0.63in."
        ),
    )
    los_sample_step: float = Field(
        gt=0,
        default=0.25,
        description=(
            "Spacing, in inches, between the points a sight line is tested at. "
            "The board is continuous, so sight is sampled rather than walked "
            "cell by cell, and this is the resolution guarantee: a blocker "
            "thinner than this can fall between two samples and leak sight. "
            "Smaller is more faithful and slower — cost is linear in it."
        ),
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
        """Backward compatibility: accept 'size' or 'width'/'height' in YAML/dict.

        The legacy keys are *consumed*, not just copied. They are aliases rather
        than fields, so leaving them behind would trip `extra="forbid"` — and
        before that existed they were simply ignored, which is the failure this
        model now rejects. Works on a copy; the caller's dict is not touched.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)
        size = data.pop("size", None)
        width = data.pop("width", None)
        height = data.pop("height", None)
        if (
            size is not None
            and "board_width" not in data
            and "board_height" not in data
        ):
            data["board_width"] = size
            data["board_height"] = size
        if width is not None and "board_width" not in data:
            data["board_width"] = width
        if height is not None and "board_height" not in data:
            data["board_height"] = height
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
        """Validate terrain outlines are in-bounds and non-overlapping.

        Checked on the resolved shapes rather than on whichever form was
        authored, so a rectangle and an outline are held to the same rule.
        Touching is not overlapping — adjacent cell rectangles share an edge once
        the board is continuous, and rejecting that would reject layouts that are
        plainly fine.
        """
        if self.terrain is None:
            return self
        polygons = [piece.to_polygon() for piece in self.terrain]
        for i, polygon in enumerate(polygons):
            x0, y0, x1, y1 = polygon.bounds
            if x0 < 0 or y0 < 0 or x1 > self.board_width or y1 > self.board_height:
                raise ValueError(
                    f"terrain[{i}] {polygon.bounds} is outside "
                    f"the board ({self.board_width}x{self.board_height})"
                )
        for i in range(len(polygons)):
            for j in range(i + 1, len(polygons)):
                if polygons[i].overlaps(polygons[j]):
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
        # Placement is done on the *bounding box*, whatever shape ends up inside
        # it, so the box is what has to fit -- an inscribed outline covers only
        # about 65% of its box but still reserves the whole thing. Using the
        # outline's own area here would let the sampler be handed specs it
        # cannot place, which fails deep inside a training run instead of at
        # load, and that is the failure this validator exists to prevent.
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
