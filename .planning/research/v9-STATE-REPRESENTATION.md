# v9.0 Research: Structured Game State & LLM-Readable Representation

**Milestone:** v9.0
**Researched:** 2026-05-23
**Focus:** Current state representation, serialisation patterns, gaps

---

## 1. Full Game State Inventory

### 1.1 Battle Aggregate (`domain/battle.py`)

The `Battle` class is the aggregate root. Its fields constitute the core mutable state:

| Field | Type | Description |
|-------|------|-------------|
| `_board_dimensions` | `BoardDimensions` (frozen dataclass) | width, height |
| `_player_models` | `list[WargameModel]` | All player units (mutable entity objects) |
| `_opponent_models` | `list[WargameModel]` | All opponent units |
| `_objectives` | `list[WargameObjective]` | Capture targets |
| `_deployment_zone` | `DeploymentZone` (frozen dataclass) | x_min, y_min, x_max, y_max |
| `_opponent_deployment_zone` | `DeploymentZone` | Same structure for opponent |
| `_player_vp` | `int` | Cumulative player victory points |
| `_opponent_vp` | `int` | Cumulative opponent victory points |
| `_player_vp_delta` | `int` | VP scored this step (reset each step) |
| `_opponent_vp_delta` | `int` | VP scored this step for opponent |

### 1.2 GameClock State (`domain/game_clock.py`)

| Field | Type | Description |
|-------|------|-------------|
| `_n_rounds` | `int` | Total battle rounds (typically 5) |
| `_first_player` | `PlayerSide` | Who goes first |
| `_game_phase` | `GamePhase` | setup / battle / complete |
| `_setup_idx` | `int` | Index into SETUP_PHASE_ORDER |
| `_round` | `int` | Current battle round (1-based) |
| `_active_player` | `PlayerSide` | Whose turn it is |
| `_phase_idx` | `int` | Index into BATTLE_PHASE_ORDER |
| `_total_steps` | `int` | Clock-level step counter |

The clock exposes a `GameState` snapshot (frozen dataclass):
- `game_phase: GamePhase`
- `setup_phase: SetupPhase | None`
- `battle_round: int | None`
- `active_player: PlayerSide | None`
- `phase: BattlePhase | None`

### 1.3 WargameModel Entity (`domain/entities.py`)

| Field | Type | Description |
|-------|------|-------------|
| `location` | `np.ndarray (2,)` | Current grid position [x, y] |
| `previous_location` | `np.ndarray \| None` | Position before last move |
| `stats` | `dict[str, int]` | Keys: `max_wounds`, `current_wounds`, `toughness`, `save` |
| `distances_to_objectives` | `np.ndarray (n_obj, 2)` | Delta vectors to each objective |
| `group_id` | `int` | Group membership |
| `previous_closest_objective_distance` | `float \| None` | For reward shaping |
| `best_closest_objective_distance` | `float \| None` | Best seen this episode |
| `model_rewards_history` | `list[ModelRewards]` | Reward breakdown per step |
| `advanced_this_turn` | `bool` | Whether model moved this turn (affects shooting) |

**Derived property:**
- `is_alive` → `stats["current_wounds"] > 0`

### 1.4 WargameObjective Entity (`domain/entities.py`)

| Field | Type | Description |
|-------|------|-------------|
| `location` | `np.ndarray (2,)` | Grid position |
| `radius_size` | `int` | Capture radius |

### 1.5 Value Objects (`domain/value_objects.py`)

- `BoardDimensions(width, height)` — frozen dataclass
- `DeploymentZone(x_min, y_min, x_max, y_max)` — frozen dataclass, has `as_array()` → `np.ndarray`

### 1.6 Combat RNG State (on `WargameEnv`)

| Field | Type | Description |
|-------|------|-------------|
| `_combat_rng` | `np.random.Generator` | Re-seeded per episode |
| `_last_player_shooting_results` | `list[ShootingResult]` | Per-step combat outcomes |
| `_last_opponent_shooting_results` | `list[ShootingResult]` | Per-step combat outcomes |

`ShootingResult` (frozen dataclass): `hits`, `wounds`, `unsaved`, `damage_dealt`

### 1.7 Env-Level Ephemeral State

| Field | Type | Description |
|-------|------|-------------|
| `current_turn` | `int` | Step counter (0-based, incremented in step()) |
| `last_reward` | `float \| None` | Most recent reward value |
| `last_reward_breakdown` | `dict[str, float]` | Per-calculator reward breakdown |
| `episode_reward_breakdown` | `dict[str, float]` | Cumulative episode reward breakdown |
| `episode_reward_steps` | `int` | Steps counted this episode |
| `last_step_context` | `StepContext \| None` | Last step's derived context |
| `_player_side` | `PlayerSide` | Which side the RL agent controls |
| `_skip_phases` | `frozenset[BattlePhase]` | Phases to auto-advance |
| `phase_manager` | `RewardPhaseManager` | Current phase state |

### 1.8 Configuration (`WargameEnvConfig` — Pydantic)

Fully serialisable, includes:
- Board dimensions, movement params (n_angles, n_speeds, max_speed)
- Entity counts and per-entity configs (`ModelConfig`, `ObjectiveConfig`)
- Deployment zones
- Reward phases (curriculum), terminal bonuses
- Skip phases, turn order, battle rounds
- Mission config (type + params)
- Opponent policy config
- `blocking_mask` (terrain)
- `max_turns_override`, `terminate_on_player_elimination`

---

## 2. Existing Serialisation

### 2.1 `WargameEnvInfo.model_dump()` (returned as Gym info dict)

The info dict exported from `step()` and `reset()` contains:

```python
{
    "current_turn": int,
    "wargame_models": [
        {
            "location": np.ndarray,        # (2,)
            "distances_to_objectives": np.ndarray,  # (n_obj, 2)
            "group_id": int,
            "max_groups": int,
            "alive": float,                # 1.0 or 0.0
            "current_wounds": int,
            "max_wounds": int,
            "weapon_attacks": int,
            "weapon_ballistic_skill": int,
            "weapon_strength": int,
            "weapon_ap": int,
            "weapon_damage": int,
            "toughness": int,
            "save_stat": int,
        },
        ...
    ],
    "objectives": [{"location": np.ndarray}],
    "opponent_models": [...],              # same schema as wargame_models
    "deployment_zone": (int, int, int, int),
    "opponent_deployment_zone": (int, int, int, int),
    "player_vp": int,
    "opponent_vp": int,
    "player_vp_delta": int,
    "opponent_vp_delta": int,
}
```

**Note:** `model_dump()` on a Pydantic model with `arbitrary_types_allowed=True` preserves numpy arrays as-is — they are NOT converted to JSON-safe lists.

### 2.2 Config Serialisation in `train.py`

- `env_config.model_dump()` → nested dict logged to Wandb as config
- `dqn_config.model_dump()` / `ppo_config.model_dump()` → same
- `to_yaml_str(env_config)` via `pydantic-yaml` in `EnvConfigCallback` → YAML file alongside checkpoints

### 2.3 Other Dict/JSON Exports

- `test_reward_phases.py` uses `cfg.model_dump()` for round-trip validation
- No explicit JSON export (no `model_dump_json()` usage found)
- No `model_json_schema()` usage found anywhere in codebase

---

## 3. Gaps: What's Missing from Info Dict vs Full State

### 3.1 Game Timing (Partially Missing)

| Data | In Info? | In Observation? | Notes |
|------|----------|-----------------|-------|
| `battle_round` | NO | YES (obs) | Info dict lacks this |
| `battle_phase` / index | NO | YES (obs) | Info dict lacks this |
| `game_phase` (setup/battle/complete) | NO | NO | Never exposed |
| `active_player` | NO | NO | Never exposed |
| `n_rounds` | NO | YES (obs) | Info dict lacks this |
| `max_turns` | NO | NO | Only on env |
| `total_clock_steps` | NO | NO | Only on GameClock |

### 3.2 Terrain

| Data | In Info? | Notes |
|------|----------|-------|
| `blocking_mask` (LOS terrain) | NO | Only in config; not in per-step output |
| LOS between specific cells | NO | Computed on demand via `has_line_of_sight_between_cells` |

### 3.3 Actions Taken

| Data | In Info? | Notes |
|------|----------|-------|
| Player action this step | NO | Not stored after application |
| Opponent action this step | NO | Applied and discarded |
| Action mask (valid actions) | NO (in obs) | In observation only |
| Which models moved/shot | NO | `advanced_this_turn` on model but not serialised |

### 3.4 Combat / Shooting Results

| Data | In Info? | Notes |
|------|----------|-------|
| `_last_player_shooting_results` | NO | `ShootingResult` dataclasses on env |
| `_last_opponent_shooting_results` | NO | Same |
| Per-model hits/wounds/damage | NO | Lost after step |

### 3.5 Reward Breakdown

| Data | In Info? | Notes |
|------|----------|-------|
| `last_reward` | NO | On env only |
| `last_reward_breakdown` | NO | Dict on env, not serialised |
| `episode_reward_breakdown` | NO | Accumulated dict on env |
| Current reward phase name | NO | On phase_manager |
| Current reward phase index | NO | On phase_manager |

### 3.6 Derived Spatial Data

| Data | In Info? | Notes |
|------|----------|-------|
| `DistanceCache` | NO | Computed each step, discarded |
| Objective ownership (per-objective) | NO | Computed by VP calc on demand |
| Group cohesion distances | NO | Computed by reward calc on demand |
| `previous_closest_objective_distance` | NO | On model entity, not in obs/info |
| `best_closest_objective_distance` | NO | Same |

### 3.7 Episode Context

| Data | In Info? | Notes |
|------|----------|-------|
| Episode seed | NO | Only used to seed `_combat_rng` |
| Turn order this episode | NO | `_player_side` on env |
| Which side is the RL agent | NO | Implicit |
| Is terminated (this step) | NO | Returned as separate Gym value |

---

## 4. Pydantic Model Assessment

### 4.1 Current Pydantic Usage

| Model | Layer | Usage |
|-------|-------|-------|
| `WargameEnvConfig` | types | Full config — YAML load/dump, Wandb logging, `model_dump()` |
| `WargameEnvInfo` | types | Info dict returned from step/reset — `model_dump()` |
| `ModelConfig` | types | Per-model YAML config |
| `ObjectiveConfig` | types | Per-objective YAML config |
| `WeaponProfile` | types | Weapon stats in config |
| `OpponentPolicyConfig` | types | Opponent policy selection |
| `MissionConfig` | types | VP calculator selection |
| `RewardPhaseConfig` | reward | Phase curriculum config |
| `RewardCalculatorConfig` | reward | Per-calculator config |
| `SuccessCriteriaConfig` | reward | Success criteria config |
| `ModelRewards` | reward/types | Per-model reward breakdown |

### 4.2 Capabilities Available but Unused

1. **`model_json_schema()`** — Not used anywhere. Could auto-generate JSON Schema for the state model, enabling LLM tool-use / function-calling integration.

2. **`model_dump_json()`** — Not used. Would produce JSON strings directly (more efficient than `model_dump()` → `json.dumps()`).

3. **`model_validate()`** / `model_validate_json()` — Only used implicitly via constructor. Could be used for state restoration/deserialization.

4. **Computed fields (`@computed_field`)** — Not used. Could expose derived properties (like `is_alive`) in serialised output without storing them.

5. **Custom serializers (`@field_serializer`)** — Not used. Would solve the numpy-to-list conversion problem for JSON-safe output.

6. **Discriminated unions** — Not used. Could model polymorphic state cleanly (e.g., different phase states).

### 4.3 Leverageability Assessment

**YES, Pydantic is the right tool for the canonical state model.** Reasons:

1. **Already the config standard** — Team is familiar, tooling (pydantic-yaml) already present
2. **Schema generation** — `model_json_schema()` produces JSON Schema that LLMs can consume as tool descriptions
3. **Validation at construction** — Ensures state integrity; catches bugs at serialisation boundaries
4. **Serialisation modes** — `model_dump(mode="json")` converts numpy arrays to lists automatically; `model_dump_json()` for direct JSON string output
5. **Nested models compose** — A `FullGameState` model can include `BoardState`, `ModelState[]`, `ClockState`, `CombatLog` etc.
6. **Backward compat** — Can annotate optional fields to grow the schema without breaking consumers

**Caveats:**
- numpy arrays need `ConfigDict(arbitrary_types_allowed=True)` or custom validators to handle JSON serialisation cleanly
- Large arrays (blocking_mask 50×50) should probably use a compact encoding rather than nested lists
- The `WargameModel` entity class is NOT Pydantic (plain class with numpy) — a parallel Pydantic `ModelState` would be needed for serialisation

---

## 5. StepContext Analysis

`StepContext` (dataclass with `slots=True`) is assembled in `WargameEnv.step()` with per-step derived data:

| Field | Type | What It Represents |
|-------|------|--------------------|
| `distance_cache` | `DistanceCache` | All spatial relationships this step |
| `current_turn` | `int` | Step number (env-level) |
| `max_turns` | `int` | Episode step limit |
| `board_width` | `int` | Board dimensions |
| `board_height` | `int` | Board dimensions |
| `is_terminated` | `bool` | Whether episode ended this step |
| `current_round` | `int` | Battle round from clock |
| `battle_phase` | `BattlePhase` | Current phase |
| `player_damage_dealt` | `int` | Damage from player shooting this step |
| `opponent_damage_dealt` | `int` | Damage from opponent shooting this step |
| `player_models_killed` | `int` | Opponent models killed this step |
| `opponent_models_killed` | `int` | Player models killed this step |

**What should be part of state output:**
- `current_round` and `battle_phase` → Game clock state (high priority)
- `player_damage_dealt` / `opponent_damage_dealt` → Combat summary (high priority for LLM narration)
- `player_models_killed` / `opponent_models_killed` → Casualty report (high priority)
- `is_terminated` → Episode status (already returned by Gym API, but useful in state blob)
- `distance_cache` → Spatial relationships (derived; maybe expose as "models at objectives" boolean array rather than raw cache)
- `max_turns` → Context for time pressure (medium priority)

---

## 6. Recommendations for v9.0 Canonical State Model

### 6.1 Proposed Structure

```python
class GameStateSnapshot(BaseModel):
    """Complete, serialisable game state for one step."""

    # -- Timing --
    step: int
    battle_round: int
    battle_phase: BattlePhase
    active_player: PlayerSide
    game_phase: GamePhase

    # -- Board --
    board_width: int
    board_height: int
    terrain: TerrainState | None  # blocking mask if present

    # -- Entities --
    player_models: list[ModelSnapshot]
    opponent_models: list[ModelSnapshot]
    objectives: list[ObjectiveSnapshot]

    # -- Zones --
    deployment_zone: tuple[int, int, int, int]
    opponent_deployment_zone: tuple[int, int, int, int]

    # -- Scoring --
    player_vp: int
    opponent_vp: int
    player_vp_delta: int
    opponent_vp_delta: int
    objective_control: list[str]  # "player" | "opponent" | "contested" | "none"

    # -- Combat (this step) --
    combat_results: CombatSummary | None

    # -- Actions --
    player_action: list[int] | None  # action indices taken
    action_descriptions: list[str] | None  # human-readable

    # -- Reward --
    reward: float | None
    reward_breakdown: dict[str, float]
    reward_phase: str

    # -- Status --
    is_terminated: bool
    termination_reason: str | None
```

### 6.2 Key Design Decisions

1. **Pydantic BaseModel** — Get `model_dump()`, `model_dump_json()`, `model_json_schema()` for free
2. **numpy → list conversion** — Use `mode="json"` or custom `@field_serializer` to produce JSON-safe output
3. **Flat where possible** — Avoid deep nesting that confuses LLMs; prefer `player_vp` over `scoring.player.vp`
4. **Optional combat/action fields** — Only populated when relevant (None during movement phase for combat results)
5. **Human-readable derivations** — Include `action_descriptions` and `termination_reason` for LLM consumption
6. **Schema export** — Use `GameStateSnapshot.model_json_schema()` to generate the tool parameter schema

### 6.3 What Needs Building

| Component | Current State | Work Needed |
|-----------|--------------|-------------|
| `GameStateSnapshot` Pydantic model | Does not exist | Create with all fields |
| numpy serialisation | `arbitrary_types_allowed` workaround | Add `@field_serializer` for clean JSON |
| Action logging | Actions applied and discarded | Store action in step before applying |
| Combat result capture | `ShootingResult` not serialised | Include in state output |
| Objective ownership | Computed on demand | Compute and include per-step |
| Reward breakdown | Dict on env, not serialised | Include in state output |
| Clock state exposure | Via `game_clock_state` property | Already accessible, just not serialised |
| Terrain state | Only in config | Compact representation in state (or reference) |
| Termination reason | `is_terminated` bool only | Add enum/string for reason |
| Human-readable action | No decoder exposed | Map action int → "move NE speed 4" / "shoot target 2" |

---

## 7. Serialisation Format Comparison

| Format | Pros | Cons | Recommendation |
|--------|------|------|----------------|
| JSON (via `model_dump_json()`) | Universal, LLM-native, schema support | Verbose for large arrays | **Primary format** |
| YAML (via `pydantic-yaml`) | Human-readable, config already uses it | Slower, less LLM tooling support | Config only |
| MessagePack | Compact, fast | Not human-readable, no LLM support | Not recommended |
| Protobuf | Schema-driven, compact | Complex setup, poor LLM interop | Not recommended |

**Verdict:** JSON with Pydantic schema generation. LLMs work natively with JSON. The `model_json_schema()` output can serve as the function parameter schema for tool-use APIs.

---

## 8. Confidence Assessment

| Finding | Confidence | Rationale |
|---------|------------|-----------|
| Full state inventory | HIGH | Direct code inspection |
| Gaps analysis | HIGH | Compared info dict to all state sources |
| Pydantic leverageability | HIGH | Features verified in Pydantic v2 docs |
| Recommended structure | MEDIUM | Design opinion; needs validation against actual LLM tool-use patterns |
| numpy serialisation approach | HIGH | Pydantic v2 `mode="json"` handles this |
