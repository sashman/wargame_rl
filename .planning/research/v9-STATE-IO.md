# v9 Research: Structured Game State & Bidirectional State I/O

**Researched:** 2026-05-23
**Confidence:** HIGH (based entirely on codebase analysis)

---

## 1. How `Battle` Is Currently Constructed

### Path: Config → Battle

```
WargameEnvConfig (Pydantic YAML)
  └─ battle_factory.from_config(config) → Battle
       ├─ _build_models(n, model_configs, n_objectives, max_groups) → list[WargameModel]
       │     - Locations initialised to np.zeros(2) — **not placed yet**
       │     - Stats: max_wounds, current_wounds (= max_wounds), toughness, save
       │     - distances_to_objectives: np.zeros([n_objectives, 2]) — **placeholder**
       │     - group_id: from config or computed from index
       ├─ _build_objectives(config) → list[WargameObjective]
       │     - Locations initialised to np.zeros(2) — **not placed yet**
       │     - radius_size: per-objective or global
       ├─ BoardDimensions(width, height)
       ├─ DeploymentZone (player, opponent)
       └─ Battle(board_dimensions, player_models, opponent_models,
                 objectives, deployment_zone, opponent_deployment_zone)
             - _player_vp = 0, _opponent_vp = 0
             - _player_vp_delta = 0, _opponent_vp_delta = 0
```

**Key observation:** `Battle` is a structural shell at construction time. Entities have zero-locations and zero-distances. Real state comes from placement (in `reset()`).

### `WargameEnv.__init__()` additionally creates:

- `ActionHandler` (config-derived, immutable after init)
- `GameClock(n_rounds=config.number_of_battle_rounds)`
- `RewardPhaseManager` (from reward_phases config — stateful, tracks curriculum phase)
- `VPCalculator` (from mission config)
- `OpponentPolicy` (if opponents configured)
- `_combat_rng` (re-seeded each episode)
- Bookkeeping: `current_turn`, `last_reward`, `last_reward_breakdown`, etc.

---

## 2. How `reset()` Initialises State

The `reset()` method follows this exact sequence:

```
1. super().reset(seed=seed)              → sets self.np_random (Gym-managed RNG)
2. Re-seed combat RNG                    → self._combat_rng = default_rng(seed)
3. Clear shooting results
4. Reset bookkeeping:
   - current_turn = 0
   - last_reward = None
   - last_step_context = None
   - last_reward_breakdown = {}
   - episode_reward_breakdown = {}
   - episode_reward_steps = 0
5. self._battle.reset_for_episode()      → clears VP, resets all model episode state
6. self._resolve_player_side()           → sets _player_side (deterministic or random)
7. self._game_clock.reset()              → setup phase, round 1, step 0
8. self._game_clock.skip_setup()         → transition to battle (round 1, player_1, command)
9. place_for_episode(battle, config, rng) → sets locations on all entities
10. run_until_player_phase(...)           → auto-execute opponent if they go first
11. compute_distances + build_observation → first obs
12. renderer.setup + render (if renderer)
```

### What `Battle.reset_for_episode()` does:

- Zeroes VP counters (player, opponent, deltas)
- Calls `model.reset_for_episode()` on every player and opponent model

### What `WargameModel.reset_for_episode()` does:

- `previous_location = None`
- `previous_closest_objective_distance = None`
- `best_closest_objective_distance = None`
- `current_wounds = max_wounds` (restore to full health)
- `model_rewards_history.clear()`
- `advanced_this_turn = False`

### Complete list of state that must be valid after reset:

| Component | State Set By | Required |
|-----------|-------------|----------|
| `np_random` | `super().reset(seed=)` | Yes (Gym contract) |
| `_combat_rng` | seeded from np_random | Yes (shooting determinism) |
| `current_turn` | `= 0` | Yes |
| `_player_side` | `_resolve_player_side()` | Yes |
| `_game_clock` | `.reset()` then `.skip_setup()` | Yes |
| Model locations | `place_for_episode()` | Yes |
| Objective locations | `place_for_episode()` | Yes |
| Model wounds | `reset_for_episode()` | Yes |
| Model distances_to_objectives | `compute_distances()` | Yes (observation) |
| VP counters | `reset_for_episode()` | Yes |
| `last_reward` / `last_step_context` | `= None` | Yes (sentinel) |
| Reward breakdown accumulators | `= {}` / `= 0` | Yes |
| Shooting results | `= []` | Yes |

---

## 3. What "Inject State" Would Mean

### 3a. Constructing a Battle from a Snapshot

A state snapshot must provide enough to construct a **mid-episode** `Battle` (or `WargameEnv`) without going through the normal `reset()` → `place_for_episode()` flow.

**Minimum fields for entity-level state injection:**

```python
# Per player/opponent model:
{
    "location": [x, y],          # int32 pair, within board bounds
    "current_wounds": int,       # 0 ≤ cw ≤ max_wounds
    "max_wounds": int,           # from config
    "group_id": int,
    "toughness": int,
    "save": int,
}

# Per objective:
{
    "location": [x, y],
    "radius_size": int,
}

# Battle-level:
{
    "player_vp": int,
    "opponent_vp": int,
}

# Clock state:
{
    "game_phase": "battle",       # or "setup" / "complete"
    "battle_round": int,          # 1..n_rounds
    "active_player": "player_1",  # or "player_2"
    "phase": "movement",          # BattlePhase enum value
    "total_steps": int,           # optional for correctness
}

# Env-level:
{
    "current_turn": int,
    "player_side": "player_1",    # which side the RL agent controls
}
```

### 3b. Can we bypass placement?

**Yes.** The current architecture already supports this conceptually:

1. `Battle` models start at `np.zeros(2)` after `from_config()`
2. `place_for_episode()` just sets `.location` on each entity
3. `fixed_wargame_model_placement()` already does direct location assignment

So bypassing placement is just: create the `Battle` from config, then directly set locations on each entity.

### 3c. Invariants That Must Hold

1. **Locations on grid**: `0 ≤ x < board_width`, `0 ≤ y < board_height` (int32)
2. **No duplicate locations** (placement logic prevents this, but not enforced elsewhere)
3. **Wounds**: `0 ≤ current_wounds ≤ max_wounds`
4. **VP**: non-negative integers
5. **Clock consistency**: `battle_round ∈ [1, n_rounds]`, `phase ∈ BattlePhase`, `active_player ∈ PlayerSide`
6. **Game phase consistency**: if `game_phase == "battle"`, all battle fields must be set
7. **Entity counts match config**: n_models, n_objectives, n_opponent_models
8. **distances_to_objectives**: must be recomputed from locations (derived, not injected)
9. **Action handler**: immutable, derived from config — no injection needed
10. **Reward history per model**: `previous_closest_objective_distance` and `best_closest_objective_distance` are reward-shaping state; should be `None` for injected states (or computable from the snapshot)

### 3d. Derived State (Recomputable, Don't Inject)

These should be recomputed after injection, not stored in the snapshot:

- `distances_to_objectives` (per model) — computed by `compute_distances()`
- `DistanceCache` — computed each step
- `action_mask` — computed each observation build
- `previous_location` — only needed for rendering, `None` is safe
- `model_rewards_history` — empty list for injected states
- VP deltas — can be zeroed (they're per-step)

---

## 4. GameClock Initialisation and Reset

### Construction

```python
GameClock(n_rounds=5, first_player=PlayerSide.player_1)
```

Sets: `_game_phase = GamePhase.setup`, `_setup_idx = 0`, `_round = 1`, `_phase_idx = 0`, `_total_steps = 0`.

### Reset

`clock.reset()` → identical to construction state (setup phase, round 1, step 0).

### Can It Be Set to an Arbitrary Phase/Round?

**Not directly.** The clock has no `set_state()` method. Its internal fields are all private (`_round`, `_phase_idx`, `_game_phase`, `_active_player`).

**Options for arbitrary state injection:**

1. **Add a `set_state()` method** (recommended):
   ```python
   def set_state(self, game_phase, round, active_player, phase_idx, total_steps):
       self._game_phase = game_phase
       self._round = round
       self._active_player = active_player
       self._phase_idx = phase_idx
       self._total_steps = total_steps
   ```

2. **Replay advances** (fragile, O(n) in rounds×phases):
   Call `skip_setup()` then `advance_phase()` repeatedly until reaching the target state.

3. **Direct attribute assignment** (hacky, breaks encapsulation):
   ```python
   clock._round = 3
   clock._phase_idx = 1  # movement
   ```

**Recommendation:** Add a `GameClock.from_state()` classmethod or `set_state()` method. It should validate the state is consistent (round in range, phase_idx in range, game_phase matches).

### Internal Fields to Set

| Field | Type | Notes |
|-------|------|-------|
| `_game_phase` | `GamePhase` | `setup`, `battle`, or `complete` |
| `_round` | `int` | 1..n_rounds |
| `_active_player` | `PlayerSide` | `player_1` or `player_2` |
| `_phase_idx` | `int` | 0..4 (index into BATTLE_PHASE_ORDER) |
| `_total_steps` | `int` | cumulative steps |
| `_n_rounds` | `int` | immutable after init (from config) |
| `_first_player` | `PlayerSide` | immutable after init |
| `_second_player` | `PlayerSide` | derived from first_player |

---

## 5. Mutable vs Construction-Time State

### Mutable During an Episode

| Entity | Field | Mutated By |
|--------|-------|-----------|
| `WargameModel` | `location` | `ActionHandler.apply()` |
| `WargameModel` | `previous_location` | `ActionHandler.apply()` |
| `WargameModel` | `stats["current_wounds"]` | `take_damage()` |
| `WargameModel` | `distances_to_objectives` | `update_distances_to_objectives()` |
| `WargameModel` | `previous_closest_objective_distance` | reward calculator |
| `WargameModel` | `best_closest_objective_distance` | reward calculator |
| `WargameModel` | `model_rewards_history` | reward calculator |
| `WargameModel` | `advanced_this_turn` | charge phase tracking |
| `Battle` | `_player_vp` / `_opponent_vp` | `add_player_vp()` / `add_opponent_vp()` |
| `Battle` | `_player_vp_delta` / `_opponent_vp_delta` | VP scoring, reset each step |
| `WargameEnv` | `current_turn` | `step()` |
| `WargameEnv` | `last_reward` | `step()` |
| `WargameEnv` | `last_step_context` | `step()` |
| `WargameEnv` | `episode_reward_breakdown` | `step()` |
| `WargameEnv` | `_last_player_shooting_results` | `step()` |
| `WargameEnv` | `_last_opponent_shooting_results` | `step()` |
| `GameClock` | `_round`, `_phase_idx`, `_active_player`, `_game_phase` | `advance_phase()` etc. |

### Fixed at Construction (Immutable During Episode)

| Entity | Field | Set By |
|--------|-------|--------|
| `WargameModel` | `stats["max_wounds"]` | `battle_factory._build_models()` |
| `WargameModel` | `stats["toughness"]` | `battle_factory._build_models()` |
| `WargameModel` | `stats["save"]` | `battle_factory._build_models()` |
| `WargameModel` | `group_id` | `battle_factory._build_models()` |
| `WargameObjective` | `radius_size` | `battle_factory._build_objectives()` |
| `Battle` | `_board_dimensions` | constructor |
| `Battle` | `_deployment_zone` / `_opponent_deployment_zone` | constructor |
| `GameClock` | `_n_rounds` | constructor |
| `GameClock` | `_first_player` / `_second_player` | constructor |
| `WargameEnv` | `config` | constructor |
| `WargameEnv` | `_action_handler` | constructor |
| `WargameEnv` | `phase_manager` | constructor (stateful for curriculum, but phase index is training-session-level, not episode-level) |

### Special: Training-Session-Level State (Not Per-Episode)

| Entity | Field | Notes |
|--------|-------|-------|
| `RewardPhaseManager` | `_current_phase_index` | Curriculum phase — advances across episodes, not within |
| `RewardPhaseManager` | `_consecutive_above` | Counter for phase advancement threshold |

---

## 6. Validation Requirements for a Valid State

### Hard Constraints (Must Fail Loudly if Violated)

1. **Model locations in bounds**: `0 ≤ x < board_width`, `0 ≤ y < board_height`, dtype int32
2. **Objective locations in bounds**: same as models
3. **Wound bounds**: `0 ≤ current_wounds ≤ max_wounds`
4. **VP non-negative**: `player_vp ≥ 0`, `opponent_vp ≥ 0`
5. **Clock phase valid**: `phase_idx` in `[0, len(BATTLE_PHASE_ORDER))`
6. **Clock round valid**: `round` in `[1, n_rounds]` when `game_phase == battle`
7. **Entity count match**: model/objective lists match config counts
8. **Board dimensions match config**: width/height from snapshot must equal env config

### Soft Constraints (Warn But Allow)

1. **No overlapping model locations**: placement prevents this, but mid-game movement could theoretically overlap (clipping doesn't check). Not currently enforced.
2. **VP delta consistency**: deltas should be ≤ cap_per_turn (but mid-step state could have any value)
3. **Objective locations outside deployment zones**: placement enforces this, but it's a placement rule, not a game rule

### Constraints That Don't Apply to Injected State

1. **Group max distance**: this is a placement constraint, not a game-state invariant
2. **`previous_closest_objective_distance`**: can be `None` (signals first step for reward shaping)

---

## 7. How Tests Set Up State — Patterns for Constructing Specific Game States

### Pattern 1: Config with Fixed Positions (Most Common)

Tests create `WargameEnvConfig` with explicit `models=[ModelConfig(x=..., y=...)]` and `objectives=[ObjectiveConfig(x=..., y=...)]`, then call `env.reset()`. This uses `fixed_wargame_model_placement()` internally.

```python
# From test_wounds.py
config = WargameEnvConfig(
    board_width=20, board_height=20,
    number_of_wargame_models=2,
    models=[ModelConfig(x=5, y=5, max_wounds=2), ModelConfig(x=10, y=10, max_wounds=2)],
    objectives=[ObjectiveConfig(x=10, y=10, radius_size=2)],
)
env = WargameEnv(config=config)
env.reset(seed=42)
```

### Pattern 2: Post-Reset Mutation (Common for Testing Specific States)

Tests call `env.reset()` first, then directly mutate entity state:

```python
# Moving models to specific positions after reset
env.reset()
env.wargame_models[0].location = np.array([5, 5])
env.objectives[0].location = np.array([10, 10])

# Inflicting damage
env.wargame_models[0].take_damage(2)

# Adding VP
env._battle.add_player_vp(20)
```

This pattern is used extensively in `test_reward_phases.py` (location mutation) and `test_wounds.py` (damage + elimination).

### Pattern 3: Standalone Entity Construction (Unit Tests)

Tests create `WargameModel` directly for isolated unit tests:

```python
# From test_wounds.py
def _make_model(max_wounds, current_wounds=None):
    return WargameModel(
        location=np.array([0, 0], dtype=np.int32),
        stats={"max_wounds": max_wounds, "current_wounds": cw},
        distances_to_objectives=np.zeros((1, 2), dtype=np.int32),
        group_id=0,
    )
```

### Pattern 4: SimpleNamespace Fakes (Reward Testing)

Tests use `SimpleNamespace` to mock `BattleView` for reward calculators:

```python
# From test_reward_phases.py
view = SimpleNamespace(player_vp_delta=5, config=SimpleNamespace(mission=...))
```

### Missing Pattern: No Tests Construct Mid-Episode State

No test currently:
- Sets the GameClock to a specific round/phase
- Injects a complete state snapshot into a running env
- Creates a Battle from a serialised representation

This confirms v9 is genuinely new capability.

---

## 8. Architectural Recommendations for State I/O

### 8a. State Representation Schema

A `GameState` dataclass/Pydantic model for serialisation:

```python
class GameSnapshot:
    board_width: int
    board_height: int

    player_models: list[ModelSnapshot]  # location, wounds, group_id, stats
    opponent_models: list[ModelSnapshot]
    objectives: list[ObjectiveSnapshot]  # location, radius

    player_vp: int
    opponent_vp: int

    clock: ClockSnapshot  # game_phase, round, active_player, phase, total_steps
    current_turn: int
    player_side: str  # "player_1" or "player_2"
```

### 8b. Serialization (State → Snapshot)

**Straightforward.** All mutable state is directly readable via properties or public attributes:
- `battle.player_models[i].location`, `.stats["current_wounds"]`, etc.
- `battle.player_vp`, `battle.opponent_vp`
- `game_clock.state` (returns `GameState` dataclass)
- `env.current_turn`

No private internals need exposure for serialisation — `GameClock.state` already returns an immutable `GameState` snapshot.

### 8c. Deserialization (Snapshot → State)

**This is the hard part.** The current design has no "set state" path. Steps needed:

1. **`GameClock.set_state()` or `from_state()`**: new method to set internal clock position
2. **`Battle.set_vp()`** or direct injection: set VP counters
3. **Model state injection**: set location, wounds on existing model objects
4. **Objective location injection**: set locations on existing objective objects
5. **`WargameEnv.load_state(snapshot)`**: orchestration method that:
   - Sets clock state
   - Sets entity positions/wounds
   - Sets VP
   - Recomputes derived state (distances, distances_to_objectives)
   - Clears per-step accumulators (reward breakdown, shooting results)
   - Sets `current_turn`
   - Builds initial observation

### 8d. Two Levels of State Injection

| Level | Use Case | What Gets Set |
|-------|----------|--------------|
| **Entity-level** | "Place these models here" | Locations, wounds only |
| **Full snapshot** | "Resume from this exact game state" | Everything: clock, VP, wounds, locations, turn count |

Entity-level injection is a subset of full snapshot — implement full snapshot and entity-level comes for free.

### 8e. Validation Layer

A `validate_snapshot(snapshot, config)` function that checks all hard constraints from section 6. Should be called before applying the snapshot.

---

## 9. Risks and Concerns

### 9a. Reward Shaping State Discontinuity

`previous_closest_objective_distance` and `best_closest_objective_distance` are used by `ClosestObjectiveCalculator` for reward shaping. When injecting mid-episode state, these will be `None`, meaning the first step after injection will return 0 reward (as if it were the first step of an episode). This is actually the safest behaviour — the alternative (computing from snapshot) would require knowing what the "previous" state was.

### 9b. Opponent Policy State

The opponent policy is stateless (selects actions based on current board state only). No concerns for state injection.

### 9c. ActionHandler State

`ActionHandler` is fully derived from config and immutable. No concerns.

### 9d. RewardPhaseManager State

The curriculum phase index is a training-session-level concern, not a per-episode concern. State injection doesn't need to touch it.

### 9e. RNG State

For full reproducibility after injection, the `_combat_rng` and `np_random` states would need to be serialised. For practical purposes (LLM-readable state, scenario setup), RNG state can be omitted and re-seeded.

### 9f. Clock `total_steps` Tracking

`GameClock._total_steps` is incremented during phase advances but isn't used for any game logic — it's purely diagnostic. For injected states, setting it to 0 or to a computed value is fine.

---

## 10. Summary: What Needs to Change

### New Code Needed

1. **`GameClock.set_state()`** — method to set clock to arbitrary position
2. **`GameSnapshot` schema** — Pydantic model for the full state representation
3. **`Battle.from_snapshot()` or `Battle.apply_snapshot()`** — inject entity state into Battle
4. **`WargameEnv.load_state(snapshot)`** — top-level injection orchestrator
5. **`WargameEnv.to_snapshot()`** — serialise current state
6. **`validate_snapshot()`** — validation against config

### No Changes Needed

- `ActionHandler` (immutable, config-derived)
- `RewardPhaseManager` (session-level state)
- `VPCalculator` (stateless computation)
- `OpponentPolicy` (stateless)
- `DistanceCache` / `compute_distances()` (recomputed each step)
- `BattleView` protocol (already read-only, works as-is)
- Observation builder (already uses BattleView)

### Existing Patterns to Leverage

- `fixed_wargame_model_placement()` — precedent for direct location setting
- `WargameModel.reset_for_episode()` — precedent for resetting per-episode state
- `GameClock.state` property — already produces `GameState` snapshot (read side)
- `Battle.reset_for_episode()` — precedent for clearing episode state
- Test patterns (Pattern 2: post-reset mutation) — shows direct mutation works
