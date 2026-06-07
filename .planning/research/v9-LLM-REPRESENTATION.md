# v9.0 Research: Structured Game State & LLM-Readable Representation

**Milestone:** v9.0 — Structured Game State & LLM-Readable Representation
**Researched:** 2026-05-23
**Confidence:** HIGH (primary source: direct codebase analysis)

## Executive Summary

This document maps the current internal representations of actions, observations, rewards, and combat results in the wargame_rl codebase, and identifies what an LLM evaluator would need to interpret the RL agent's behaviour. The goal is to enable an LLM to read a structured step summary and judge whether the trained RL model is making good or bad decisions.

The codebase has rich structured data at every step — action indices that can be decoded to human-readable descriptions, full combat narratives in `ShootingResult`, detailed reward breakdowns, and game timing state. However, none of this is currently surfaced as text. The Pygame renderer's tooltip and south panel are the closest to human-readable output, but they only show a fraction of available data and are visual-only.

---

## 1. Action Encoding & Decoding

### Action Space Structure

Actions are encoded as a single integer per model, managed by `ActionHandler` and partitioned by `ActionRegistry` into contiguous slices:

| Slice | Indices | Valid Phases | Meaning |
|-------|---------|--------------|---------|
| `stay` | `0` | All phases | No action (pass) |
| `movement` | `1 .. N×S` | Movement only | Polar coordinate move |
| `shooting` | `N×S+1 .. N×S+T` | Shooting only | Target selection (conditional on opponents existing) |

**Default configuration:** `n_movement_angles=16`, `n_speed_bins=6` → 96 movement actions. With 4 opponents: 101 total (1 stay + 96 movement + 4 shooting).

### Movement Action Decoding

Movement actions encode `(angle_idx, speed_idx)` pairs:

```
action 0          → STAY (dx=0, dy=0)
action 1..N×S     → move_idx = action - 1
                     angle_idx = move_idx // n_speed_bins
                     speed_idx = move_idx % n_speed_bins
```

**Angles:** `n_movement_angles` evenly-spaced directions starting at 0 rad (east/+x), counter-clockwise:
- With 16 angles: 0°, 22.5°, 45°, 67.5°, 90° (N), 112.5°, 135°, 157.5°, 180° (W), 202.5°, 225°, 247.5°, 270° (S), 292.5°, 315°, 337.5°

**Speeds:** Linearly spaced from `max_move_speed / n_speed_bins` up to `max_move_speed`:
- With `max_move_speed=6, n_speed_bins=6`: speeds = [1, 2, 3, 4, 5, 6]

**Displacement:** The continuous (angle × speed) vector is rounded to nearest integer cell. Pre-computed in `_displacements[angle_idx, speed_idx]` → `(dx, dy)` as int.

### Shooting Action Decoding

```
target_idx = action - shooting_slice.start
```

Target index K maps to opponent model K in the observation. Positional alignment is a critical invariant.

### Human-Readable Action Description

To convert an action integer to text, the LLM representation layer needs:

```
action 0 → "Stay (no action)"
action 1-96 → "Move {direction_name} at speed {speed} (dx={dx}, dy={dy})"
action 97-100 → "Shoot at opponent model {target_idx}"
```

**Key method:** `ActionHandler._decode_action(action) → np.ndarray([dx, dy])` already exists. For LLM narration, we also need the angle/speed decomposition from `encode_action`/decoding logic, and compass direction mapping.

**What exists today:**
- `_decode_action()` returns `(dx, dy)` displacement
- `encode_action(angle_idx, speed_idx)` encodes back
- `best_action_toward(dx, dy)` finds closest action to a direction
- No human-readable text generation exists

**What's needed for LLM:**
- `describe_action(action_int, model_idx, phase) → str` that produces e.g. "Model 0 moves NE at speed 4 (dx=3, dy=3)" or "Model 2 shoots at opponent 1"
- Phase context matters: same action index means different things in movement vs shooting phase

---

## 2. Observation Structure

### `WargameEnvObservation` (what the agent sees)

| Field | Type | Description |
|-------|------|-------------|
| `current_turn` | int | Step counter |
| `wargame_models` | list[WargameModelObservation] | Player model states |
| `objectives` | list[WargameEnvObjectiveObservation] | Objective locations |
| `board_width` / `board_height` | int | Board dimensions |
| `opponent_models` | list[WargameModelObservation] | Opponent model states |
| `action_mask` | np.ndarray (n_models, n_actions) | Valid action mask |
| `battle_round` | int | Current round (1-based) |
| `battle_phase_index` | int | Phase index (0=command, 1=movement, 2=shooting, 3=charge, 4=fight) |
| `n_rounds` | int | Total rounds |
| `player_vp` / `opponent_vp` | int | Cumulative victory points |
| `player_vp_delta` | int | VP gained this step |

### `WargameModelObservation` (per-model)

| Field | Type | Description |
|-------|------|-------------|
| `location` | np.ndarray (2,) | (x, y) grid position |
| `distances_to_objectives` | np.ndarray | Delta vectors to each objective |
| `group_id` | int | Unit group assignment |
| `max_groups` | int | Max groups (for one-hot) |
| `alive` | float | 1.0 alive, 0.0 dead |
| `current_wounds` / `max_wounds` | int | Wound pool |
| `weapon_attacks` | int | Number of hit rolls per shot |
| `weapon_ballistic_skill` | int | D6 roll needed to hit (e.g. 3 = 3+) |
| `weapon_strength` | int | For wound roll comparison |
| `weapon_ap` | int | Armour penetration |
| `weapon_damage` | int | Wounds per failed save |
| `toughness` | int | Defensive stat |
| `save_stat` | int | Base armour save |

### `WargameEnvObjectiveObservation`

| Field | Type | Description |
|-------|------|-------------|
| `location` | np.ndarray (2,) | (x, y) grid position |

### `WargameEnvInfo` (extra info dict)

Superset of observation data plus:
- `deployment_zone` / `opponent_deployment_zone`: (x_min, y_min, x_max, y_max) tuples
- `player_vp_delta` / `opponent_vp_delta`: VP changes for both sides

### Tensor Pipeline

`observation_to_tensor()` produces 5 tensors:
1. **game_features** `(6,)`: placeholder, normalized_round, normalized_phase, player_vp, opponent_vp, player_vp_delta
2. **objectives** `(n_obj, 2)`: normalized locations
3. **player_models** `(n_models, feature_dim)`: normalized location, distances, group one-hot, closest same-group distance, alive, wound_ratio, max_wounds, 7 combat stats, expected_damage per opponent
4. **opponent_models** `(n_opponents, feature_dim)`: same structure + zero padding
5. **action_mask** `(n_models, n_actions)`: boolean

The tensor pipeline computes `expected_damage` per player-opponent pair using the analytical formula — this is already an interpretable combat metric.

---

## 3. Step Summary Requirements for LLM Evaluator

An LLM evaluator needs a structured summary per step. Based on the data available:

### State Before Action

```
Round: {battle_round}/{n_rounds}  Phase: {phase_name}  Step: {current_turn}/{max_turns}
VP: Player {player_vp} | Opponent {opponent_vp}

Player Models:
  Model 0 (group 0): ({x}, {y}) — {current_wounds}/{max_wounds} HP — weapon: {attacks}A BS{bs}+ S{str} AP-{ap} D{dmg} range {range}"
  Model 1 (group 0): ({x}, {y}) — {current_wounds}/{max_wounds} HP — DEAD
  ...

Opponent Models:
  Model 0 (group 0): ({x}, {y}) — {current_wounds}/{max_wounds} HP — weapon: ...
  ...

Objectives:
  Objective 0: ({x}, {y}) — radius {r} — controlled by: {player/opponent/contested/none}

Distances: Model 0 → Obj 0: {dist}, Model 0 → Obj 1: {dist}, ...
```

### Action Taken (decoded)

```
Actions:
  Model 0: Move NE at speed 4 (dx=3, dy=3) — new position: ({x'}, {y'})
  Model 1: Stay (dead, forced)
  Model 2: Shoot at opponent model 1
  Model 3: Move E at speed 6 (dx=6, dy=0) — new position: ({x'}, {y'})
```

### Combat Results (shooting phase)

The `ShootingResult` dataclass provides full narrative:

```python
@dataclass(frozen=True, slots=True)
class ShootingResult:
    hits: int       # successful hit rolls
    wounds: int     # successful wound rolls
    unsaved: int    # wounds that got past armour
    damage_dealt: int  # total damage inflicted
```

Plus combat resolution context:
- Wound roll threshold computed from `weapon.strength` vs `defender.toughness`
- Modified save = `defender.save + weapon.ap`
- Damage = `unsaved × weapon.damage`

**Narrative example:**
```
Model 2 fires at Opponent 1 (range 8, weapon: 2A BS3+ S4 AP-1 D1):
  Hit rolls: 2 attacks → 1 hit (need 3+)
  Wound rolls: 1 → 1 wound (S4 vs T3, need 3+)
  Save rolls: 1 → 0 saved (4+ save, AP-1 = need 5+)
  Result: 1 unsaved wound → 1 damage dealt
  Opponent 1: 2/3 → 1/3 HP
```

**Currently stored:** `_last_player_shooting_results` and `_last_opponent_shooting_results` on `WargameEnv` — lists of `ShootingResult`, one per model that fired. These are available during the step but not persisted into observations or info.

**Gap:** Individual dice rolls are NOT stored — only aggregate counts (hits, wounds, unsaved, damage_dealt). The RNG is used in `resolve_shooting()` but results are summarized. For full dice narration, we'd need to either store the rolls or re-derive probabilities.

### Reward Breakdown

`WargameEnv` already tracks:
- `last_reward`: float total reward
- `last_reward_breakdown`: dict[str, float] with keys like `"closest_objective"`, `"group_cohesion"`, `"vp_gain"`, `"terminal_success_bonus"`, sub-components like `"closest_objective/distance_delta"`, `"closest_objective/base_penalty"`, `"closest_objective/best_distance_bonus"`
- `episode_reward_breakdown`: cumulative across episode

**Narrative example:**
```
Reward: -0.450
  closest_objective: -0.300 (avg across models)
    distance_delta: +0.015 (got further from objective)
    base_penalty: -0.330
    best_distance_bonus: 0.000
  group_cohesion: -0.150 (models too far apart, excess 3.0 beyond max_distance 5.0)
  vp_gain: 0.000 (no VP scored this step)
```

### State After

Same format as "State Before" with updated positions, wounds, VP.

### Opponent Turn Summary

The opponent's full turn is auto-executed between player steps. Currently this is invisible — the player just sees the result in the next observation. For LLM evaluation, the opponent's actions and results should also be narrated.

**Gap:** Opponent actions are applied but not stored. `_apply_opponent_action()` calls the policy and applies the action, but doesn't record what action was taken. Only opponent shooting results are stored in `_last_opponent_shooting_results`.

---

## 4. Existing Human-Readable Output

### Pygame Renderer South Panel

Displays (text rendered on screen, not logged):
- Round / Phase / Step
- Reward (3 decimal places)
- Player VP (+delta) | Opponent VP (+delta)
- Epoch (if set)

### Pygame Tooltip (hover/click on model)

Shows for player models:
- Location: (x, y)
- Group ID
- Closest objective reward (from `ModelRewards`)
- Group distance violation penalty
- Total reward

**Note:** `ModelRewards` is a legacy Pydantic model that only tracks `closest_objective_reward` and `group_distance_violation_penalty`. It doesn't reflect the full reward breakdown from the phase manager.

### Logging

Loguru is used but primarily for:
- Reward phase advancement: "Reward phase advanced: 'reach_objectives' -> 'win_by_vp' (success_rate=0.95, epoch=42)"
- No per-step logging of actions, observations, or combat results

### What's Missing

| Data | Available Internally | Surfaced to User |
|------|---------------------|-----------------|
| Action description (decoded) | `_decode_action()` exists | No |
| Movement direction name | Angles computed | No |
| Shooting target + result | `ShootingResult` stored per step | No |
| Wound roll math | `wound_roll_threshold()` exists | No |
| Reward breakdown | `last_reward_breakdown` dict | Tooltip (partial, legacy) |
| Opponent actions | Applied but not stored | No |
| VP scoring reason | Computed in `_on_before_advance` | No |
| Objective control | Computed for rendering | No (visual only) |
| Phase description | `BattlePhase` enum | South panel (visual) |
| Game clock state | Full `GameState` | Round/phase in south panel |

---

## 5. Game Concepts Needing Natural Language Descriptions

### Battle Phases

```python
class BattlePhase(str, Enum):
    command = "command"      # VP scoring happens here (end of phase, round 2+)
    movement = "movement"    # Models move using polar-coordinate actions
    shooting = "shooting"    # Models select targets and resolve attacks
    charge = "charge"        # Future: melee charge declarations
    fight = "fight"          # Future: melee combat resolution
```

Default `skip_phases` skips all non-movement phases. When shooting is enabled, command/charge/fight are typically skipped.

### Objectives & Control

- Objectives are circular zones with a `radius_size` (in grid cells)
- A model is "at" an objective if its distance (offset by radius) ≤ 0
- **Control:** determined by `objective_ownership_from_norms_offset()` — the side with at least one alive model within radius controls the objective; if both sides have models, it's contested (neither controls)
- Control is computed for rendering (fill color) and VP scoring but not included in observations directly

### VP Scoring (Mission System)

- **Default mission:** Score VP at the end of the command phase from round 2+
- `vp_per_objective` (default 5) × number of controlled objectives, capped at `cap_per_turn` (default 15)
- VP is cumulative; tracked as `player_vp`, `opponent_vp`
- `player_vp_delta` is the VP gained in the current step (observable to the agent)
- Success criteria `player_vp_min` checks if player VP at end of episode ≥ fraction of theoretical max

### Weapon Profiles

Each model can have weapons with these stats:

| Stat | Field | Description | Example |
|------|-------|-------------|---------|
| Range | `range` | Max range in grid cells | 12 |
| Attacks | `attacks` | Number of hit dice | 2 |
| Ballistic Skill | `ballistic_skill` | D6 roll needed to hit | 3 (= 3+) |
| Strength | `strength` | For wound comparison | 4 |
| AP | `ap` | Worsens target save | 1 |
| Damage | `damage` | Wounds per unsaved | 1 |

### Shooting Resolution Sequence

Full attack sequence per model (tabletop rules):
1. **Hit Roll:** Roll `attacks` D6s. Hit on `ballistic_skill`+. Nat 1 always fails, nat 6 always hits.
2. **Wound Roll:** Roll `hits` D6s. Threshold from strength vs toughness:
   - S ≥ 2×T → 2+ | S > T → 3+ | S = T → 4+ | S < T → 5+ | T ≥ 2×S → 6+
3. **Save Roll:** Roll `wounds` D6s. Target needs `save + AP`+ to save. Nat 1 always fails.
4. **Damage:** `unsaved × damage` total damage applied to target's wound pool.

### Wounds & Elimination

- Models have `current_wounds` / `max_wounds`
- When `current_wounds` hits 0, model is dead (`is_alive → False`)
- Dead models are forced to `STAY_ACTION` only
- Damage applied via `take_damage(amount)` — clamped to 0, no overkill tracking

### Deployment Zones

- Rectangular areas `(x_min, y_min, x_max, y_max)` where models spawn
- Player zone typically left side, opponent zone right side
- Models are placed randomly within zone (or at fixed positions if configured)

### Turn Order

- `player` / `opponent` / `random` (coin-flip each reset)
- Within a round, both sides take all phases in sequence
- Opponent's full turn (all phases) is auto-executed between player steps

---

## 6. Combat Narrative Data (`ShootingResult`)

### What Exists

```python
@dataclass(frozen=True, slots=True)
class ShootingResult:
    hits: int          # number of successful hit rolls
    wounds: int        # number of successful wound rolls
    unsaved: int       # wounds that penetrated armour
    damage_dealt: int  # total damage = unsaved × weapon.damage
```

**Stored per step on `WargameEnv`:**
- `_last_player_shooting_results: list[ShootingResult]` — one per player model that fired
- `_last_opponent_shooting_results: list[ShootingResult]` — one per opponent model that fired

**Available but not stored:**
- Which model fired at which target (action → target_idx mapping in `_resolve_shooting_action`)
- Individual dice rolls (consumed by numpy, only aggregates kept)
- Wound threshold used (computable from `wound_roll_threshold(weapon.strength, defender.toughness)`)
- Modified save value (computable from `defender.save + weapon.ap`)

### Expected Damage (Analytical)

`expected_damage(weapon, defender) → float` provides closed-form expected damage. Already computed per player-opponent pair in the tensor pipeline as an observation feature. This is directly useful for LLM evaluation: "Model 0 has expected damage of 0.67 against Opponent 1 vs 1.33 against Opponent 0 — choosing Opponent 1 was suboptimal."

### What's Missing for Full Narration

1. **Attacker-target pairing per result:** Currently `_last_player_shooting_results` is a flat list. Need to store `(attacker_idx, target_idx, result)` tuples.
2. **Dice roll details:** Individual rolls are consumed by numpy and not stored. For LLM narration, either store rolls or reconstruct probabilities. Storing probabilities (hit chance, wound threshold, modified save) is more useful than individual dice.
3. **Opponent action recording:** Opponent shooting results exist but opponent movement actions are not stored.

---

## 7. Reward System Details for LLM Interpretation

### Reward Calculators (what generates the signal)

| Calculator | Type | What it rewards/penalizes |
|------------|------|--------------------------|
| `closest_objective` | per-model | Penalty when model doesn't get closer to nearest objective. 0 when improving. Optional bonus for new best distance. |
| `group_cohesion` | per-model | Penalty proportional to excess distance beyond `group_max_distance` from nearest same-group model. 0 when within range. |
| `vp_gain` | global | Reward = player_vp_delta / cap_per_turn. Max 1.0 per step when scoring full cap. |

### Terminal Bonuses

- `terminal_success_bonus`: Added at episode end when all models at objectives. Scaled by remaining turns fraction (faster = bigger bonus).
- `terminal_vp_bonus`: Added at episode end when player VP meets threshold.

### Breakdown Structure

`last_reward_breakdown` dict keys:
- `"closest_objective"` → weighted average across models
- `"closest_objective/distance_delta"` → sub-component average
- `"closest_objective/base_penalty"` → sub-component average
- `"closest_objective/best_distance_bonus"` → sub-component average
- `"group_cohesion"` → weighted average across models
- `"vp_gain"` → global value
- `"terminal_success_bonus"` → one-time bonus at termination
- `"terminal_vp_bonus"` → one-time bonus at termination

### Curriculum Phases

The reward function changes during training via `RewardPhaseManager`. An LLM evaluator needs to know which phase is active to interpret rewards correctly:

```
Phase 0 "reach_objectives": closest_objective(1.0) + group_cohesion(0.5)
  → Evaluate: Is model moving toward objectives while staying grouped?

Phase 1 "win_by_vp": vp_gain(1.0) + closest_objective(0.1) + group_cohesion(0.5)
  → Evaluate: Is model scoring VP? Still roughly moving toward objectives?

Phase 2 "win_only_by_vp": vp_gain(1.0) + group_cohesion(0.5)
  → Evaluate: Is model maximizing VP score?
```

---

## 8. Implementation Recommendations

### Step Summary Data Structure

A new structured step summary should capture:

```python
@dataclass
class StepSummary:
    # Timing
    step: int
    battle_round: int
    n_rounds: int
    phase: str  # "movement" / "shooting"

    # Scores
    player_vp: int
    opponent_vp: int
    player_vp_delta: int
    opponent_vp_delta: int

    # Actions taken (decoded)
    player_actions: list[ActionDescription]  # per-model
    opponent_actions: list[ActionDescription] | None  # when available

    # Combat results
    player_shooting: list[CombatNarrative]
    opponent_shooting: list[CombatNarrative]

    # Reward
    reward: float
    reward_breakdown: dict[str, float]
    reward_phase_name: str

    # State snapshot (before/after or just after)
    player_models: list[ModelSnapshot]
    opponent_models: list[ModelSnapshot]
    objectives: list[ObjectiveSnapshot]

@dataclass
class ActionDescription:
    model_idx: int
    action_int: int
    action_type: str  # "stay" / "move" / "shoot"
    description: str  # "Move NE at speed 4 (dx=3, dy=3)" or "Shoot at opponent 1"
    # Movement-specific
    direction: str | None  # "NE", "E", etc.
    speed: int | None
    displacement: tuple[int, int] | None
    # Shooting-specific
    target_idx: int | None

@dataclass
class CombatNarrative:
    attacker_idx: int
    target_idx: int
    weapon_profile: str  # "2A BS3+ S4 AP-1 D1"
    hit_chance: float  # probability
    wound_threshold: int  # e.g. 3 for "3+"
    modified_save: int  # e.g. 5 for "5+"
    result: ShootingResult  # hits, wounds, unsaved, damage_dealt
    target_wounds_before: int
    target_wounds_after: int
    target_eliminated: bool
```

### Text Serialization

For LLM consumption, serialize as structured text (not JSON — too verbose). Example:

```
=== Step 15 | Round 3/5 | Movement Phase ===
VP: Player 10 | Opponent 5

Player Actions:
  Model 0 (group 0, at 25,12, 3/3 HP): Move E at speed 5 → (30,12)
  Model 1 (group 0, at 22,14, 3/3 HP): Move ENE at speed 4 → (25,12)
  Model 2 (group 1, at 18,30, 2/3 HP): Move NE at speed 6 → (22,26)
  Model 3 (group 1, at 20,28, 3/3 HP): Move NE at speed 3 → (22,26)

Reward: -0.150
  closest_objective: -0.100 (models 0,1 getting closer; model 2 slightly further)
  group_cohesion: -0.050 (group 1 models 2.0 beyond max distance 5.0)

Objectives: Obj 0 at (30,6) [player controlled] | Obj 1 at (30,22) [contested] | Obj 2 at (30,38) [opponent controlled]
```

### Key Data Sources for Each Component

| Summary Component | Source in Codebase | Gap |
|---|---|---|
| Action description | `ActionHandler._decode_action()` + angle/speed decomposition | Need text formatting layer |
| Model state | `WargameModelObservation` or `WargameModel` | Available, need serialization |
| Objective control | `objective_ownership_from_norms_offset()` | Computed for render, not stored in obs |
| Shooting result | `ShootingResult` + env fields | Need attacker-target pairing |
| Reward breakdown | `last_reward_breakdown` dict | Available, need text formatting |
| VP scoring | `_on_before_advance()` + `VPCalculator` | Reasons not stored |
| Opponent actions | `_apply_opponent_action()` | Actions not recorded |
| Expected damage | `expected_damage()` in tensor pipeline | Available, good LLM signal |
| Wound math | `wound_roll_threshold()` | Available as function |

### Implementation Priority

1. **Action decoder to text** — simplest, most impactful. Add `describe_action()` method to `ActionHandler`.
2. **Step summary data structure** — new file in `envs/types/` or `envs/narration/`.
3. **Store attacker-target pairing** — modify `_resolve_shooting_action` to return `(model_idx, target_idx, result)` tuples.
4. **Record opponent actions** — store the `WargameEnvAction` from opponent policy before applying it.
5. **Objective control in observations** — add to `WargameEnvInfo` or summary.
6. **Text serialization** — format the step summary as structured text for LLM consumption.

### Architectural Notes

- The LLM representation layer should depend on `BattleView` (read-only protocol), not `WargameEnv` directly.
- Step summaries should be optional — don't slow down training. Enable via config flag or only during evaluation/simulation.
- Consider a `StepNarrator` class that takes `BattleView` + `StepContext` + decoded actions and produces text.
- The reward phase name from `RewardPhaseManager.current_phase_name` is essential context for the LLM — without it, reward values are uninterpretable.

---

## 9. Compass Direction Mapping

For human-readable movement narration, map angle indices to compass directions:

| Angles (16) | Direction | Degrees |
|-------------|-----------|---------|
| 0 | E (East) | 0° |
| 1 | ENE | 22.5° |
| 2 | NE | 45° |
| 3 | NNE | 67.5° |
| 4 | N (North) | 90° |
| 5 | NNW | 112.5° |
| 6 | NW | 135° |
| 7 | WNW | 157.5° |
| 8 | W (West) | 180° |
| 9 | WSW | 202.5° |
| 10 | SW | 225° |
| 11 | SSW | 247.5° |
| 12 | S (South) | 270° |
| 13 | SSE | 292.5° |
| 14 | SE | 315° |
| 15 | ESE | 337.5° |

**Note:** The grid convention is +x = East, +y = South (screen coordinates). The angle system starts at East and goes counter-clockwise, so angle_idx=4 (90°) is actually North (-y direction in screen coords). The narration layer needs to handle this coordinate system correctly.

---

## 10. Summary of Gaps

| Gap | Severity | Fix Complexity |
|-----|----------|---------------|
| No text representation of actions | High | Low — decode + format |
| Shooting results lack attacker-target pairing | High | Low — modify return type |
| Opponent actions not recorded | Medium | Low — store before applying |
| Objective control not in observations | Medium | Low — add to info dict |
| No step summary data structure | High | Medium — new types + assembly |
| Individual dice rolls not stored | Low | Medium — optional recording |
| Reward phase context not in step output | Medium | Low — add phase name to info |
| VP scoring reasons not recorded | Low | Low — log controlled count |
| Coordinate system documentation | Medium | Low — document in narration code |
