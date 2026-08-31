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
from wargame_rl.wargame.envs.types.config.coherency import CoherencyConfig
from wargame_rl.wargame.envs.types.config.entities import ModelConfig, ObjectiveConfig
from wargame_rl.wargame.envs.types.config.melee import MeleeConfig
from wargame_rl.wargame.envs.types.config.terrain import (
    MapPoolConfig,
    RandomTerrainConfig,
    TerrainPieceConfig,
)
from wargame_rl.wargame.envs.types.game_timing import (
    MELEE_ONLY_PHASES,
    NON_MOVEMENT_PHASES,
    BattlePhase,
)

# The rules' infantry base is 32mm across, so its radius is 16mm = 0.63".
#
# Authored here rather than imported from `domain/rules_constants.py`, which
# holds the same number: `types/` is the shared kernel and cannot import
# `domain/` without inverting the dependency direction (see docs/ddd-envs.md).
# `tests/test_rules_constants.py` pins the two together so they cannot drift.
INFANTRY_BASE_RADIUS_IN = 32.0 / 25.4 / 2.0


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
    map_pool: MapPoolConfig | None = Field(
        default=None,
        description="Draw a whole layout — terrain and objectives — from a set "
        "of fixed maps, one per episode. The third terrain mode, mutually "
        "exclusive with both `terrain` and `random_terrain`, and the only one "
        "that trains on real tables. A pool whose maps differ in objective or "
        "piece count needs `objective_budget` / `terrain_budget` to match, which "
        "is checked when the pool is loaded. None = no pool.",
    )
    track_exposure: bool = Field(
        default=False,
        description="Accumulate line-of-sight exposure and terrain-proximity "
        "statistics during shooting phases. Measurement only — it does not affect "
        "the game, but it costs an extra shooting-mask build per shooting phase.",
    )
    track_opponent_coherency: bool = Field(
        default=False,
        description="Accumulate the OPPONENT force's unit-coherency totals as "
        "well as the player's. Measurement only -- it changes no outcome -- but "
        "it costs one extra coherency evaluation per opponent movement phase, "
        "so it is off by default and switched on by the code that needs the "
        "column. A rated leg needs it: `evaluate_selector` measures the player "
        "seat, so an entrant seated only as B would otherwise carry no coherency "
        "figure, and a vp_margin without one is a result plus an unstated claim "
        "that the moves earning it were legal.",
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
    observe_coherency: bool = Field(
        default=False,
        description="Put the two halves of the coherency rule the observation "
        "never carried on each model's token, widening it by two: the distance "
        "to the *furthest* live model in its unit over the spread cap, and the "
        "fraction of the unit in its own chain component. The existing "
        'same-group column is a *nearest* neighbour distance, so the 9" spread '
        "condition had no tensor at all and a unit strung across the board read "
        "as tight from every model in it; connectivity is a transitive closure "
        "and is not recoverable from pairwise distances cheaply. Both are "
        "normalised by the coherency distances rather than the board diagonal — "
        "against the diagonal the whole 2\" band is 2.7% of a column's range, so "
        "the decision-relevant region is compressed into noise. This is the "
        "input any coherency reward or enforcement must key on; adding it first "
        "is the desk check this project has twice paid ~10 GPU-hours to skip. "
        "Default False keeps the tensor byte-identical; turning it on changes "
        "the per-model embedding shape, so existing checkpoints fail to load — "
        "the intended loud failure. UNTRAINED.",
    )
    observe_unit_centroid: bool = Field(
        default=False,
        description="Put the vector from each model to its unit's live centroid "
        "on that model's token, widening it by two (dx, dy). "
        "**`observe_coherency` carries magnitudes; this carries direction.** A "
        "model can already be told its unit is stretched or split, and still "
        "have nothing saying which way to move to fix it — the spread ratio is "
        "the same number whichever side of the unit it sits on. "
        "Added because the strongest scripted policies keep formation "
        "*structurally* rather than by choosing to: `squad_march` moves every "
        "model of a unit along ONE shared centroid vector, so relative "
        "positions are preserved by construction. A behaviour clone at 98.6% "
        "action match reproduces that only to 0.665 unit coherency against the "
        "demonstrator's 0.884, and cloning from referee-corrected "
        "demonstrations measured null — so the hypothesis is that the pattern "
        "is not *representable* from the current inputs rather than merely "
        "unlearned. This is the quantity the demonstrator actually computes. "
        "Normalised by the spread cap, not the board diagonal, for the reason "
        "given above, and clipped per axis so the sign survives at any "
        "distance — direction is the point, and `observe_coherency` already "
        "carries the magnitude. "
        "Default False keeps the tensor byte-identical; turning it on changes "
        "the per-model embedding shape, so existing checkpoints fail to load — "
        "the intended loud failure. UNTRAINED.",
    )
    observe_unit_strength: bool = Field(
        default=False,
        description="Put each model's *unit* remaining strength (alive members "
        "/ unit size) on its own token, widening the per-model token by one. "
        "Shooting names a unit and the defender allocates, so how many models "
        "a unit has left decides whether a volley finishes it or is thrown at "
        "a full one — and no input carried it: the shooting head mean-pools "
        "opponent tokens into one token per unit, and a mean is invariant to "
        "how many terms it averages. The column is constant across a unit's "
        "members, so every token states it, with no change to the pooling or "
        "the projection. Default False keeps the tensor byte-identical; turning "
        "it on changes the per-model embedding shape, so existing checkpoints "
        "fail to load — the intended loud failure. "
        "UNTRAINED, AND ITS CHEAPEST PROXY MEASURED NULL: a scripted policy "
        "firing at the *weakest* valid unit rather than the nearest scores "
        "+1.7 +/- 5.7 vp_margin paired over 100 identical layouts (t = 0.30), "
        "winning 24 of 100. An unpaired 60-episode read of the same comparison "
        "said +8.0, and that was noise. The choice is not rare — 59.5% of "
        "shooters see more than one valid unit and 72% of those see units of "
        "differing strength — it simply does not pay much here, for a reason "
        "already on record: unit targeting discards only 3.6% of declared "
        "attacks, which caps what finishing a unit early can reclaim. Turn "
        "this on only behind a mechanism that is not bounded by that 3.6%.",
    )
    objective_budget: int | None = Field(
        default=None,
        ge=1,
        description="Pad every objective-derived input to this many slots, so "
        "scenarios with different objective counts share one network. The "
        "objective token gains a trailing `present` column (1 real, 0 padding) "
        "and the per-model block gains one presence column per slot beside its "
        "padded distance pairs — without them a padding slot's zero distance "
        "reads as 'this model is standing on it'. Needed because the per-model "
        "distance block is `2 + n_objectives * 2` wide, which makes objective "
        "count a hard input dimension: the 45 real layouts carry five or six "
        "objectives, so one network cannot span them, and none can be scored by "
        "a checkpoint trained at three. None = no padding, byte-identical to a "
        "config without the field; setting it changes both embedding shapes, so "
        "existing checkpoints fail to load — the intended loud failure.",
    )
    terrain_budget: int | None = Field(
        default=None,
        ge=1,
        description="Pad the terrain token sequence to this many pieces, so "
        "layouts with different piece counts collate into one batch. Padding "
        "rows are all zero, including the vertex-count column, which no real "
        "piece can be — that is what marks them, and the network drops them from "
        "attention. The shipped maps carry 15 or 16 pieces, which "
        "`observations_to_tensor_batch` cannot stack. None = no padding.",
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
    deployment_outline: list[tuple[float, float]] | None = Field(
        default=None,
        min_length=3,
        description="Player deployment zone as an outline, replacing the "
        "rectangle. The real deployments are triangles, staircases and arcs; "
        "only two of the six are axis-aligned bands. None keeps the rectangle, "
        "which is the exact no-op.",
    )
    opponent_deployment_outline: list[tuple[float, float]] | None = Field(
        default=None,
        min_length=3,
        description="Opponent deployment zone as an outline. See `deployment_outline`.",
    )
    models: list[ModelConfig] | None = Field(
        default=None,
        description="Per-model configuration (attributes, and optionally positions). Length must match number_of_wargame_models.",
    )
    objectives: list[ObjectiveConfig] | None = Field(
        default=None,
        description="Per-objective configuration (attributes, and optionally positions). Length must match number_of_objectives.",
    )
    objectives_on_terrain: bool | None = Field(
        default=None,
        description="Make each objective *be* a terrain piece: the **largest** "
        "pieces whose centre lies in the middle section — between the two "
        "deployment edges, though a piece may overlap one — become area "
        "objectives whose outline is the footprint. This is the rules' terrain "
        "objective: the ground itself is the prize, which puts cover and the "
        "contested ground in the same place. Selection is constrained to be "
        "mirror-symmetric (fairness) and unclustered, with separation ranking "
        "within the pool the size filter hands it.\n\n"
        "Three states, because 'the author asked for this' and 'this is merely "
        "the default' need different failure behaviour:\n"
        "  None (default) — use terrain objectives when the layout can host "
        "them and the config does not place objectives itself; otherwise fall "
        "back to discs. A board with no ruins has nothing to put an objective "
        "on, and a config that hand-places its objectives has already said "
        "where it wants them; failing either would be failing it for a setting "
        "its author never made.\n"
        "  True — require them, and raise when the layout cannot deliver. "
        "Silently placing discs would turn a three-objective terrain mission "
        "into something else while looking like it worked.\n"
        "  False — always free-floating discs of `objective_radius_size`, which "
        "is what every pre-geometry result was measured under.\n\n"
        "Tri-state rather than a bool plus `model_fields_set`: that was tried "
        "and is broken, because `model_dump()` round-trips lose which fields "
        "were set and training dumps the config, so every field came back "
        "looking explicit. The distinction has to live in the value.",
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
        description=(
            "Maximum number of groups in the game; group_id is one-hot encoded "
            "over this size for neural network input. **One value for both "
            "armies**, and it is load-bearing twice over: the one-hot is the "
            "same width on the player and opponent blocks, which is what keeps "
            "their token widths equal, and where no model names its own group "
            "`group_span = n // max_groups` splits each army, so two armies of "
            "different sizes get differently-sized units from the same cap."
        ),
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
    declare_objectives: bool = Field(
        default=False,
        description=(
            "Register an OBJECTIVE-DECLARATION slice (size = objective_budget, "
            "valid in the command phase): each squad's LEADER may declare which "
            "objective the squad is committed to, the declaration binds the "
            "unit and PERSISTS until re-declared (STAY keeps the current one; "
            "no squad is ever forced to hold a plan it wants to change). The "
            "agent's own allocation plan, as a first-class action -- the "
            "learning form of the surplus-reallocation rule "
            "(baseline/reallocation.py) whose play-time decode measures "
            "+14.54 +/- 3.81 vp, and the design the four failed travel terms "
            "point at: they paid approach toward targets a heuristic imposed, "
            "where `declared_objective_progress` pays the agent for executing "
            "a commitment it chose itself (the charge_progress post-mortem: "
            "'the declaration gate is the entire difference'). False (the "
            "default) registers NO slice and adds NO observation column: every "
            "existing config keeps its exact action space and checkpoints. "
            "Turning it on is UNPAIRABLE against a config without it (new "
            "actions change the output head)."
        ),
    )
    declare_targets: bool = Field(
        default=False,
        description=(
            "Register an ENEMY-TARGET declaration slice (size = max_groups, "
            "valid in the command phase): each squad's LEADER may declare which "
            "enemy unit the squad is hunting; the declaration binds the unit "
            "and PERSISTS until re-declared (STAY keeps it). The hunt analog "
            "of declare_objectives, priced by declared_target_progress -- the "
            "movement-phase channel is the one that never reverts, which is "
            "what made the objective form trainable where the charge-phase "
            "forms (level: farmed; delta: revert-blanked) were not."
        ),
    )
    hunt_declares_charge: bool = Field(
        default=False,
        description=(
            "THE FOLD (s35's command-slot lesson): a unit whose leader has "
            "declared an enemy target auto-declares its charge in every "
            "command phase in which the charge is legal, WITHOUT spending the "
            "leader's command action -- a hunt IS charge-intent. The override "
            "lives at the plan level: re-declaring an objective drops the "
            "hunt (and vice versa, the plan is ONE commitment). Adds no "
            "actions, so it is init-PAIRABLE against the same config with it "
            "off. Requires declare_targets."
        ),
    )
    n_advance_speed_bins: int = Field(
        ge=0,
        default=0,
        description=(
            "Number of ADVANCE speed bins, appended as their own action slice. "
            "An advance trades the turn's shooting for reach: maximum distance "
            "is the model's Move PLUS an advance roll (one D6 per unit, made "
            "before moving), and a model that advances cannot shoot this turn "
            "-- no weapon here has the ability that would let it. "
            "0 (the default) registers NO slice, makes NO dice draw and adds NO "
            "observation column, so every existing config keeps its exact "
            "action space, its checkpoints and its RNG stream. "
            "⚠ The slice is appended AFTER shooting on purpose. Widening the "
            "existing movement bins instead would renumber every action, "
            "because `decode_action` is angle-major and speed-minor -- action 7 "
            "would stop meaning (angle 1, speed 0) and start meaning (angle 0, "
            "speed 6). Warm starts load with `strict=False`, so that would "
            "scramble every checkpoint silently."
        ),
    )
    # Slices named here are registered at full width but valid in NO phase, so
    # every one of their actions is masked to -inf for the whole episode.
    #
    # This exists to restore PAIRING to action-space experiments, which are
    # otherwise the least measurable class of change here: adding actions
    # widens the policy head, which changes how much RNG `seed_everything`
    # consumes, so an arm and its control no longer start from the same weights
    # and the paired estimator -- worth roughly an order of magnitude -- is
    # lost. Registering the slice in BOTH arms and darkening it in the control
    # makes the two parameter shapes identical, and their initial weights
    # bit-identical (verified in `tests/test_dark_action_slices.py`).
    #
    # ⚠ It does NOT make an existing control reusable. A control trained
    # without the slice has a different head width and therefore different
    # weights everywhere -- measured, only 73 of 110 shared-shape tensors
    # match -- so the control must be retrained WITH the slice darkened.
    dark_action_slices: list[str] = Field(
        default_factory=list,
        description=(
            "Action slices to register but never make valid, so an arm and its "
            "control share a parameter shape and can be paired."
        ),
    )

    base_radius: float = Field(
        ge=0,
        default=INFANTRY_BASE_RADIUS_IN,
        description=(
            "Radius of a model's base, in inches. Gives a model a physical "
            "extent: bases may not overlap at placement, objective range is "
            "measured from the base edge rather than its centre, models occlude "
            "sight and block movement, engagement is base to base, and the "
            "renderer draws it at this size. Defaults to the rules' infantry "
            "base, 32mm across -- a radius of 16mm, 0.63in. "
            "0.0 makes models dimensionless points again, which is what every "
            "result measured before continuous space assumed and what the "
            "default used to be: at 0.0 no disc occludes, the three cover rays "
            "coincide so cover cannot occur at all, models do not collide, and "
            "range is centre to centre. Setting it to 0.0 therefore switches "
            "off four mechanics at once, silently."
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
    coherency: CoherencyConfig = Field(
        default_factory=CoherencyConfig,
        description="The unit coherency rule (docs/rules/03-moving.md): its two "
        "distances, and which of its consequences are enforced. Every "
        "consequence defaults to off, so the default is exactly the behaviour "
        "that predates it.",
    )
    melee: MeleeConfig = Field(
        default_factory=MeleeConfig,
        description="Whether the charge and fight phases are played "
        "(docs/rules/11-charge-phase.md, 12-fight-phase.md). Defaults off, "
        "which is an exact no-op: no slice, no dice, no observation column.",
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
    def require_coherency_observation_for_coherency_reward(self) -> "WargameEnvConfig":
        """`unit_coherency` needs `observe_coherency`, so refuse the pair unset.

        The desk check from [docs/reward-phases.md](../../../../docs/reward-phases.md)
        § Design rules, made mechanical: *if two states differ only in what this
        term keys on, do they differ in the observation?* Without
        `observe_coherency` an objective's worth of coherency information never
        reaches the network -- neither the 9" spread, which is a
        furthest-neighbour quantity no existing column carries, nor
        connectivity, which is a transitive closure. The reward would then be
        unattributable, and an unattributable reward is experienced only as
        "this pays less".

        Rejected at construction because the failure is silent and expensive:
        the overstack penalty and `objective_hold.surplus_value` each burned GPU
        hours keying on per-objective counts the agent could not see.
        """
        uses_coherency_reward = any(
            calculator.type == "unit_coherency"
            for phase in self.reward_phases
            for calculator in phase.reward_calculators
        )
        if uses_coherency_reward and not self.observe_coherency:
            raise ValueError(
                "a reward phase uses the 'unit_coherency' calculator but "
                "observe_coherency is False, so the agent cannot see what the "
                "reward keys on -- set observe_coherency: true"
            )
        return self

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
    def validate_map_pool(self) -> "WargameEnvConfig":
        """The three terrain modes are mutually exclusive.

        Whichever pair were combined, one would silently win: a pool installs its
        layout at reset and would overwrite a fixed `terrain`, and `random_terrain`
        would regenerate over a drawn map — the same failure `measure-maps` guards
        against by clearing the generator, which is silent because the run still
        prints a map's name.
        """
        if self.map_pool is None:
            return self
        conflicting = [
            name
            for name, value in (
                ("terrain", self.terrain),
                ("random_terrain", self.random_terrain),
            )
            if value is not None
        ]
        if conflicting:
            raise ValueError(
                f"map_pool is mutually exclusive with {' and '.join(conflicting)}: "
                "terrain is drawn from the pool, fixed, or generated — not two of "
                "those"
            )
        return self

    @model_validator(mode="after")
    def validate_melee_needs_a_charge_phase(self) -> "WargameEnvConfig":
        """Melee needs the charge phase, where a charge is declared and made.

        The same failure `validate_advance_needs_a_command_phase` catches: the
        actions would exist and never be legal, so a training run would spend
        hours measuring a feature it never had.

        ⚠ **The FIGHT phase is deliberately NOT required, and requiring it was a
        bug.** The fight carries no agent action -- its mask offers exactly one
        legal option per model (STAY) -- and it resolves in `_on_before_advance`,
        which fires on skipped phases. So the correct melee config *skips*
        fight and steps only charge. An earlier version of this validator
        demanded both, which rejected that config and would have forced an extra
        agent step per round, +33% of episode length, for a choice with one
        option. Caught by validating a real config rather than by a test.

        ⚠ **The reverse direction is also not checked.** Rejecting a stepped
        charge or fight while melee is off was proposed, on the grounds that it
        inflates `max_turns` for nothing. But `skip_phases: []` is a *documented*
        setting for full per-phase stepping (`envs/CLAUDE.md`), five test modules
        construct it, and no shipped config steps either phase -- so the check
        would reject legitimate configs to guard against a mistake nothing has
        made.
        """
        # ⚠ **AUTO-SKIPPED WHERE MELEE IS OFF, and that is what keeps every
        # existing config a no-op.** `max_turns` is
        # `n_rounds x (len(BATTLE_PHASE_ORDER) - len(skip_phases))`, so adding
        # two phases would silently lengthen every episode in the repo. Skipping
        # them when there is no melee keeps the stepped-phase count, and
        # therefore `max_turns`, exactly as it was -- without editing dozens of
        # configs that have no opinion about a phase they never reach.
        if not self.melee.enabled:
            for phase in MELEE_ONLY_PHASES:
                if phase not in self.skip_phases:
                    self.skip_phases.append(phase)

        if self.melee.enabled and BattlePhase.charge in self.skip_phases:
            raise ValueError(
                "melee.enabled needs the charge phase, where a charge is "
                "declared and made -- remove 'charge' from skip_phases"
            )
        return self

    @model_validator(mode="after")
    def validate_advance_needs_a_command_phase(self) -> "WargameEnvConfig":
        """A scenario with advance rungs must let its units declare a move type.

        The move type is declared in the command phase, so a config that skips
        that phase can register the rungs and never make one legal -- the
        advance would be silently unavailable for the whole run. Caught here
        rather than at the first `reset`, because a training run would otherwise
        spend hours measuring a feature it never had.
        """
        if self.n_advance_speed_bins > 0 and BattlePhase.command in self.skip_phases:
            raise ValueError(
                "n_advance_speed_bins > 0 needs the command phase, where a unit "
                "declares its move type -- remove 'command' from skip_phases"
            )
        # ⚠ Melee needs it for the same reason. A charge rung is legal only for
        # a unit that declared a charge, and the declaration is made in the
        # command phase -- so a melee config that skips it steps the whole
        # charge phase with exactly one option per model, and a training run
        # would measure a mechanic it never had.
        if self.melee.enabled and BattlePhase.command in self.skip_phases:
            raise ValueError(
                "a stepped charge phase needs the command phase, where a "
                "unit's leader declares a charge and binds the unit -- remove "
                "'command' from skip_phases"
            )
        # ⚠ The objective declaration lives in the command phase too, and its
        # slice is sized to the objective BUDGET -- both must exist, or a run
        # trains a plan it can never make (or a slice of size zero).
        if self.declare_objectives:
            if BattlePhase.command in self.skip_phases:
                raise ValueError(
                    "declare_objectives needs the command phase, where a "
                    "unit's leader declares its objective -- remove 'command' "
                    "from skip_phases"
                )
            if self.objective_budget is None:
                raise ValueError(
                    "declare_objectives sizes its action slice to "
                    "objective_budget, which is unset -- set objective_budget"
                )
        if self.declare_targets:
            if BattlePhase.command in self.skip_phases:
                raise ValueError(
                    "declare_targets needs the command phase, where a unit's "
                    "leader declares its hunt -- remove 'command' from "
                    "skip_phases"
                )
            if not self.melee.enabled:
                raise ValueError(
                    "declare_targets is a hunt declaration; without "
                    "melee.enabled the charge it aims at does not exist"
                )
        if self.hunt_declares_charge and not self.declare_targets:
            raise ValueError(
                "hunt_declares_charge folds the charge into the hunt "
                "declaration, so it needs declare_targets -- without a hunt "
                "there is nothing to fold"
            )
        return self

    @model_validator(mode="after")
    def validate_observation_budgets(self) -> "WargameEnvConfig":
        """Reject a budget smaller than what this scenario actually puts on the board.

        A budget that does not fit is worse than none: the padding helpers would
        have to drop real objectives or real terrain to honour it, and the
        network would score a board it cannot see all of. Checked here rather
        than at the first `reset`, since a training run would otherwise fail an
        hour in.
        """
        if (
            self.objective_budget is not None
            and self.objective_budget < self.number_of_objectives
        ):
            raise ValueError(
                f"objective_budget ({self.objective_budget}) is below "
                f"number_of_objectives ({self.number_of_objectives})"
            )
        if self.terrain_budget is not None:
            n_pieces = len(self.terrain) if self.terrain is not None else 0
            if self.random_terrain is not None:
                n_pieces = self.random_terrain.count
            if self.terrain_budget < n_pieces:
                raise ValueError(
                    f"terrain_budget ({self.terrain_budget}) is below the "
                    f"scenario's {n_pieces} terrain pieces"
                )
        return self

    @model_validator(mode="after")
    def validate_coherency(self) -> "WargameEnvConfig":
        """Reject a coherency setting that could not bind, or could not be met.

        Two ways to switch the rule on and get nothing. The distances can be the
        wrong way round, which makes the spread cap looser than the chain it is
        supposed to bound. And the army can split into one-model units, every one
        of which is coherent by definition -- `max_groups` defaults to 100, so a
        25-model config that never sets it gets 25 units of one and the whole
        rule is a silent no-op. Both are caught here rather than discovered from
        a flat metric after a training run.
        """
        coherency = self.coherency
        if coherency.furthest_distance < coherency.nearest_distance:
            raise ValueError(
                f"coherency.furthest_distance ({coherency.furthest_distance}) is "
                f"below nearest_distance ({coherency.nearest_distance}): the "
                "spread cap must be at least the chain distance"
            )
        if not (coherency.enforce_at_deployment or coherency.attrition):
            return self
        for label, count in (
            ("number_of_wargame_models", self.number_of_wargame_models),
            ("number_of_opponent_models", self.number_of_opponent_models),
        ):
            if count > 1 and max(1, count // self.max_groups) < 2:
                raise ValueError(
                    f"coherency is enforced but {label}={count} with "
                    f"max_groups={self.max_groups} puts every model in its own "
                    "unit, where coherency holds vacuously. Set max_groups below "
                    f"{label} so units have at least two models."
                )
        return self

    @model_validator(mode="after")
    def validate_group_ids_fit_the_one_hot(self) -> "WargameEnvConfig":
        """Every declared `group_id` must be encodable in the group one-hot.

        `_group_ids_to_one_hot` *clips* to `max_groups - 1`, so a config naming
        more units than the cap allows does not fail — it silently merges its
        last units into one column, and every consumer that reads membership
        back out of the observation inherits the merge. The shooting slice does
        not clip: `unit_count` sizes it off the highest id, so the mask and the
        network would disagree about which unit an action names.
        """
        for label, configs in (
            ("models", self.models),
            ("opponent_models", self.opponent_models),
        ):
            if not configs:
                continue
            highest = max(int(config.group_id) for config in configs)
            if highest >= self.max_groups:
                raise ValueError(
                    f"{label} declares group_id={highest} but max_groups="
                    f"{self.max_groups}: the group one-hot has no column for it "
                    "and would silently fold it into the last group. Raise "
                    f"max_groups above {highest}."
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
