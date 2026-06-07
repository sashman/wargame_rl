# Phase v9-01: Canonical State Model & Export — Research

**Researched:** 2026-05-23
**Domain:** Pydantic state serialisation, game state capture, numpy→JSON conversion
**Confidence:** HIGH

## Summary

This phase creates a `GameStateSnapshot` Pydantic model that captures the full game state at any point during an episode and exports it to JSON. The codebase already has all the data internally — it just needs a parallel output path that reads from `BattleView` and domain objects, converts numpy arrays to native Python types at construction, and exposes the result as a Pydantic model.

The critical technical findings are: (1) the snapshot model **must not contain numpy arrays** — Pydantic v2's `model_json_schema()` throws `PydanticInvalidForJsonSchema` on `np.ndarray` fields even with `@field_serializer`, so all numpy data must be converted to `list[int]`/`list[float]` at construction time; (2) `ShootingResult` lacks attacker-target pairing — the `_resolve_shooting_action` method knows the mapping but discards it; (3) opponent actions are produced by `_opponent_policy.select_action()` and immediately applied without recording; (4) reward breakdown and phase name are available on `phase_manager` but not surfaced outside the env.

**Primary recommendation:** Build `GameStateSnapshot` and sub-models using only native Python types (no numpy). Convert at construction in a factory function `build_snapshot(view, ...)` that reads `BattleView` properties and `.tolist()` all arrays. Add `to_snapshot()` on `WargameEnv` as the public API.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SGS-01 | Canonical programmatic game-state model exists, sourced from domain/read-only views | §Architecture Patterns: `GameStateSnapshot` as BattleView consumer in `envs/state/` |
| SGS-04 | Default encoding is JSON; codec interface for additional formats | §Standard Stack: Pydantic v2 `model_dump_json()`; codec pattern via `Encoder` protocol |
| SGS-11 | Combat results include attacker-target pairing and analytical context | §Code Examples: `CombatResultSnapshot` with `attacker_idx`, `target_idx`; modify `_resolve_shooting_action` |
| SGS-12 | Opponent actions recorded before application and available in state output | §Code Examples: Store `opp_action` in `_apply_opponent_action` before applying |
| SGS-14 | Reward breakdown and active reward phase name included in state output | §Code Examples: Read from `phase_manager.last_reward_breakdown` and `current_phase_name` |
</phase_requirements>

## Project Constraints (from .cursor/rules/)

Directives extracted from `.cursor/rules/` and `CLAUDE.md` that constrain this phase:

- **DDD architecture** (`docs/ddd-envs.md`): New state module lives at `envs/state/`, depends on `BattleView` + `types/` only. Domain layer must not import from `state/`. [VERIFIED: ddd-envs.md]
- **BattleView protocol**: State export is a read-only consumer. Add properties to BattleView only if needed (additive, non-breaking). [VERIFIED: battle_view.py]
- **Pydantic + pydantic-yaml for structured data**: The snapshot model uses Pydantic BaseModel. [VERIFIED: codebase convention]
- **Type hints required**: All public functions typed; `from __future__ import annotations`. [VERIFIED: CLAUDE.md]
- **Backward compat**: New config fields must default to no-op. Existing YAML configs must keep working. [VERIFIED: CLAUDE.md]
- **`just validate` before pushing**: Format, lint, test must pass. [VERIFIED: CLAUDE.md]
- **Registry pattern for extensible subsystems**: If codec interface is added, use string-keyed registry matching reward/criteria/opponent/mission registries. [VERIFIED: reward/calculators/registry.py pattern]
- **Keep runtime simple**: Prefer validation at construction time, push complexity to startup. [VERIFIED: CLAUDE.md]
- **No numpy in JSON schema**: `model_json_schema()` on Pydantic models with `np.ndarray` fields raises `PydanticInvalidForJsonSchema`. Use native Python types. [VERIFIED: tested locally with Pydantic 2.11.10]

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Snapshot model definition | `envs/state/` (new) | — | State serialisation is an adapter concern, peer of `renders/` |
| Numpy→Python conversion | `envs/state/` factory | — | Conversion happens at snapshot construction time, not in domain |
| Combat result capture (attacker-target) | `envs/wargame.py` (facade) | `domain/shooting.py` (return type) | Facade orchestrates shooting; pairing info is available in `_resolve_shooting_action` |
| Opponent action recording | `envs/wargame.py` (facade) | — | Facade calls opponent policy and applies action; recording is a facade concern |
| Reward breakdown exposure | `envs/wargame.py` (facade) | `reward/phase_manager.py` | Already stored on env; snapshot reads it |
| Objective control computation | `envs/env_components/distance_cache.py` | — | `objective_ownership_from_norms_offset()` already exists; snapshot calls it |
| Schema versioning | `envs/state/` | — | Metadata field on the snapshot model |
| `to_snapshot()` public API | `envs/wargame.py` (facade) | — | Env facade assembles the snapshot from its own state |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Pydantic | 2.11.10 | Snapshot model, JSON serialisation, schema generation | Already the config/types standard; `model_dump_json()` and `model_json_schema()` for free [VERIFIED: installed] |

### Supporting

No additional packages needed. This phase uses only existing dependencies (pydantic, numpy).

## Package Legitimacy Audit

No new packages required. All dependencies are already in the project.

## Architecture Patterns

### System Architecture Diagram

```
WargameEnv.step()
  ├── [existing] build_observation(view) → tensors → RL networks
  │
  └── [NEW] to_snapshot()
        ├── reads BattleView properties (models, objectives, clock, VP)
        ├── reads env-level ephemeral state (reward, actions, shooting)
        ├── converts numpy → native Python at construction
        └── returns GameStateSnapshot (Pydantic BaseModel)
              ├── .model_dump_json() → JSON string
              ├── .model_dump() → Python dict
              └── .model_json_schema() → JSON Schema document
```

### Recommended Project Structure

```
wargame_rl/wargame/envs/
├── state/                      # NEW — snapshot models and factory
│   ├── __init__.py             # Re-exports GameStateSnapshot, build_snapshot
│   └── snapshot.py             # Pydantic models + build_snapshot() factory
├── domain/                     # UNCHANGED
├── env_components/             # UNCHANGED
├── reward/                     # UNCHANGED
├── renders/                    # UNCHANGED
├── types/                      # UNCHANGED
└── wargame.py                  # ADD: to_snapshot() method, combat pairing, opponent recording
```

Minimal file count — a single `snapshot.py` is sufficient for Phase 1. The `state/` package can grow in later phases (formatters, event log, replay).

### Pattern 1: Snapshot as BattleView Consumer

**What:** `GameStateSnapshot` is constructed by a factory function that reads from `BattleView` properties and env-level fields, converting all numpy arrays to native Python types at construction time.

**When to use:** Any time the full game state needs to be serialised (after step, after reset, on demand).

**Why not on the model itself:** The snapshot model is a pure data carrier (Pydantic BaseModel). The factory function does the adaptation work (reading BattleView, converting types). This separates the "what" (model shape) from the "how" (construction from live env state).

### Pattern 2: Native Python Types Only (No Numpy in Pydantic)

**What:** All snapshot model fields use `list[int]`, `list[float]`, `tuple[int, ...]`, etc. — never `np.ndarray`. Conversion happens in the factory.

**Why:** Pydantic v2's `model_json_schema()` throws `PydanticInvalidForJsonSchema` on `np.ndarray` fields. Using native types enables `model_dump_json()`, `model_dump()`, and `model_json_schema()` to all work without workarounds. [VERIFIED: tested locally]

### Pattern 3: Extended ShootingResult with Pairing

**What:** Introduce `CombatResultSnapshot` that pairs attacker index, target index, weapon profile, and `ShootingResult` outcome. This is a Pydantic model in `state/snapshot.py` constructed from data available inside `_resolve_shooting_action`.

**Why not modify ShootingResult directly:** `ShootingResult` is a domain value object (frozen dataclass) used by the shooting resolution engine. Adding attacker/target indices to it would leak facade-level concerns (model indexing) into the domain. Instead, the facade wraps `ShootingResult` + pairing info into a higher-level `CombatResultSnapshot`.

### Anti-Patterns to Avoid

- **numpy in Pydantic fields:** Breaks `model_json_schema()`. Always convert to native Python at construction.
- **Modifying BattleView for snapshot data:** Combat results, opponent actions, and reward breakdowns are env-level ephemeral state, not BattleView concerns. Pass them to the factory function as arguments, don't add them to the protocol.
- **Lazy serialisation:** Don't store numpy arrays and convert in a `@field_serializer`. Convert at construction time so the model is always JSON-ready. This follows the project convention of "push complexity to startup, keep runtime simple."

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| JSON serialisation | Custom `to_dict()` methods | Pydantic `model_dump_json()` | Handles nested models, optional fields, enums automatically |
| JSON Schema generation | Manual schema dict | Pydantic `model_json_schema()` | Auto-generates from type annotations; stays in sync with model |
| Numpy array conversion | Custom recursive converter | `.tolist()` at construction | Built into numpy; converts to native Python recursively |
| Objective control | Custom distance logic | `objective_ownership_from_norms_offset()` | Already exists in `distance_cache.py`; used by VP calculator and renderer |

## Common Pitfalls

### Pitfall 1: numpy in Pydantic breaks JSON Schema

**What goes wrong:** Using `np.ndarray` as a Pydantic field type causes `model_json_schema()` to throw `PydanticInvalidForJsonSchema`, even with `ConfigDict(arbitrary_types_allowed=True)` and `@field_serializer`.
**Why it happens:** Pydantic v2 needs to map every field to a JSON Schema type. `np.ndarray` has no JSON Schema equivalent.
**How to avoid:** Use `list[int]`, `list[float]`, etc. for all fields. Convert numpy arrays at snapshot construction time via `.tolist()`.
**Warning signs:** `arbitrary_types_allowed=True` in snapshot model config — this is the wrong approach.

### Pitfall 2: Stale Shooting Results After Phase Advance

**What goes wrong:** After the player's action, `run_after_player_action` may advance through opponent phases, overwriting `_last_opponent_shooting_results`. If the snapshot is captured at the wrong point, it might reflect only the final phase's results.
**Why it happens:** The env resets `_last_player_shooting_results` and `_last_opponent_shooting_results` at the start of each `step()` call, then populates them during action resolution. By the time `step()` returns, they contain the correct per-step results.
**How to avoid:** Capture the snapshot at the end of `step()` (after all action resolution and phase advancement complete). The `to_snapshot()` method reads the current state of `_last_*_shooting_results` which is correct at that point.
**Warning signs:** Snapshot showing empty combat results when shooting occurred.

### Pitfall 3: Opponent Action Not Recorded

**What goes wrong:** The opponent action is produced by `self._opponent_policy.select_action()` and immediately fed to `_opponent_action_handler.apply()` or `_resolve_shooting_action()`. After application, the action object is garbage collected.
**Why it happens:** The env was designed for RL training where opponent actions don't matter to the agent's learning signal.
**How to avoid:** Store the `WargameEnvAction` returned by `select_action()` as `self._last_opponent_action` before applying it. The snapshot reads this field.
**Warning signs:** Snapshot's `opponent_actions` always `None`.

### Pitfall 4: Numpy Scalar Types in JSON

**What goes wrong:** `np.int32(42)` is not `int` — `json.dumps()` raises `TypeError: Object of type int32 is not JSON serializable`.
**Why it happens:** Numpy scalars from array indexing retain their numpy dtype.
**How to avoid:** Use `.tolist()` on arrays (converts recursively), `int(x)` for individual scalars, or `float(x)` for floats. The factory function must handle this consistently.
**Warning signs:** JSON serialisation errors mentioning `int32`, `int64`, `float64`.

### Pitfall 5: Objective Control Requires Distance Recomputation

**What goes wrong:** Objective ownership needs both player and opponent distance caches, but the env's `DistanceCache` (computed in `step()`) only covers player models.
**Why it happens:** `compute_distances()` in `step()` is called for player models only. VP scoring computes a separate opponent cache on demand.
**How to avoid:** The snapshot factory computes opponent distances when needed, using the same `compute_distances()` and `objective_ownership_from_norms_offset()` functions. This is cheap (vectorised numpy) and matches the VP calculator's approach.
**Warning signs:** Missing or incorrect objective control in snapshot.

## Code Examples

### Snapshot Model Shape (Verified from Codebase Analysis)

```python
# Source: analysis of BattleView properties, entities, and env-level fields

class ModelSnapshot(BaseModel):
    """Serialisable snapshot of one WargameModel."""
    location: list[int]  # [x, y]
    previous_location: list[int] | None
    group_id: int
    alive: bool
    current_wounds: int
    max_wounds: int
    toughness: int
    save: int
    advanced_this_turn: bool

class ObjectiveSnapshot(BaseModel):
    """Serialisable snapshot of one WargameObjective."""
    location: list[int]  # [x, y]
    radius_size: int

class ClockSnapshot(BaseModel):
    """Serialisable snapshot of GameClock state."""
    game_phase: str      # "setup" | "battle" | "complete"
    battle_round: int | None
    active_player: str | None  # "player_1" | "player_2"
    battle_phase: str | None   # "command" | "movement" | "shooting" | ...

class CombatResultSnapshot(BaseModel):
    """One model's shooting result with attacker-target pairing."""
    attacker_idx: int
    target_idx: int
    hits: int
    wounds: int
    unsaved: int
    damage_dealt: int

class RewardSnapshot(BaseModel):
    """Reward breakdown for the current step."""
    total: float | None
    breakdown: dict[str, float]
    phase_name: str
    phase_index: int

class GameStateSnapshot(BaseModel):
    """Complete, serialisable game state at one point in time."""
    schema_version: str

    # Timing
    step: int
    max_steps: int
    clock: ClockSnapshot
    n_rounds: int

    # Board
    board_width: int
    board_height: int

    # Entities
    player_models: list[ModelSnapshot]
    opponent_models: list[ModelSnapshot]
    objectives: list[ObjectiveSnapshot]

    # Zones
    deployment_zone: tuple[int, int, int, int]
    opponent_deployment_zone: tuple[int, int, int, int]

    # Scoring
    player_vp: int
    opponent_vp: int
    player_vp_delta: int
    opponent_vp_delta: int
    objective_control: list[str]  # per-objective: "player" | "opponent" | "contested" | "none"

    # Actions taken this step
    player_actions: list[int] | None
    opponent_actions: list[int] | None

    # Combat results this step
    player_combat_results: list[CombatResultSnapshot]
    opponent_combat_results: list[CombatResultSnapshot]

    # Reward
    reward: RewardSnapshot

    # Episode status
    is_terminated: bool
    is_truncated: bool
```

### Converting WargameModel to ModelSnapshot

```python
# Source: WargameModel entity fields (domain/entities.py)

def _model_to_snapshot(model: WargameModel) -> ModelSnapshot:
    return ModelSnapshot(
        location=model.location.tolist(),
        previous_location=(
            model.previous_location.tolist()
            if model.previous_location is not None
            else None
        ),
        group_id=model.group_id,
        alive=model.is_alive,
        current_wounds=model.stats["current_wounds"],
        max_wounds=model.stats["max_wounds"],
        toughness=model.stats["toughness"],
        save=model.stats["save"],
        advanced_this_turn=model.advanced_this_turn,
    )
```

### Recording Attacker-Target Pairing

The pairing is available in `_resolve_shooting_action` but currently discarded:

```python
# Source: wargame.py lines 326-369 (_resolve_shooting_action)
# Current code iterates (i, act) and computes target_idx = act - shooting_slice.start
# but only returns ShootingResult (no pairing info).

# Fix: return list of (attacker_idx, target_idx, ShootingResult) tuples.
# Or: store pairing alongside results on the env.

# Concrete approach — add a dataclass for the paired result:
@dataclass(frozen=True, slots=True)
class PairedShootingResult:
    attacker_idx: int
    target_idx: int
    result: ShootingResult

# Modify _resolve_shooting_action to return list[PairedShootingResult]
# and store as self._last_player_shooting_results / _last_opponent_shooting_results.
```

Here is the exact code path for the pairing. In `_resolve_shooting_action`, the loop already has `i` (attacker index) and `target_idx`:

```python
# Current (wargame.py:341-348):
for i, act in enumerate(action.actions):
    # ... validity checks ...
    target_idx = act - shooting_slice.start
    # ... resolve_shooting() ...
    results.append(result)  # <-- loses i and target_idx

# Proposed:
    results.append(PairedShootingResult(
        attacker_idx=i,
        target_idx=target_idx,
        result=result,
    ))
```

### Recording Opponent Actions

The opponent action is produced and immediately applied in `_apply_opponent_action`:

```python
# Source: wargame.py:391-418 (_apply_opponent_action)
# Current:
opp_action = self._opponent_policy.select_action(...)
# immediately applied, opp_action is lost after method returns

# Fix: store before applying
self._last_opponent_action = opp_action  # ADD this line
# then apply as before
```

The stored `_last_opponent_action` is read by `to_snapshot()`.

### Objective Control Computation

```python
# Source: distance_cache.py objective_ownership_from_norms_offset()
# VP calculator already uses this. Snapshot factory reuses it.

from wargame_rl.wargame.envs.env_components.distance_cache import (
    compute_distances,
    objective_ownership_from_norms_offset,
)

def _compute_objective_control(view: BattleView) -> list[str]:
    player_alive = alive_mask_for(view.player_models)
    player_cache = compute_distances(view.player_models, view.objectives, alive_mask=player_alive)

    if view.opponent_models:
        opp_alive = alive_mask_for(view.opponent_models)
        opp_cache = compute_distances(view.opponent_models, view.objectives, alive_mask=opp_alive)
        opp_norms = opp_cache.model_obj_norms_offset
    else:
        opp_norms = np.zeros((0, len(view.objectives)), dtype=np.float64)

    p_ctrl, o_ctrl = objective_ownership_from_norms_offset(
        player_cache.model_obj_norms_offset, opp_norms, player_cache.obj_radii
    )
    # Derive per-objective label
    result = []
    for i in range(len(view.objectives)):
        p_any = np.any(player_cache.model_obj_norms_offset[:, i] <= player_cache.obj_radii[i])
        o_any = np.any(opp_norms[:, i] <= player_cache.obj_radii[i]) if opp_norms.shape[0] > 0 else False
        if p_any and o_any:
            result.append("contested")
        elif p_ctrl[i]:
            result.append("player")
        elif o_ctrl[i]:
            result.append("opponent")
        else:
            result.append("none")
    return result
```

### Codec / Encoder Interface (SGS-04)

```python
# Minimal encoder protocol for future format extensibility.
# Phase 1 only implements JSON; the protocol enables Phase 4 additions.

class SnapshotEncoder(Protocol):
    """Encodes a GameStateSnapshot to a specific format."""

    def encode(self, snapshot: GameStateSnapshot) -> str | bytes: ...
    def content_type(self) -> str: ...

class JsonEncoder:
    """Default JSON encoder using Pydantic's built-in serialisation."""

    def encode(self, snapshot: GameStateSnapshot) -> str:
        return snapshot.model_dump_json(indent=2)

    def content_type(self) -> str:
        return "application/json"
```

### to_snapshot() on WargameEnv

```python
# Source: architecture analysis — env facade assembles snapshot from its state

def to_snapshot(self) -> GameStateSnapshot:
    """Return a serialisable snapshot of the current game state."""
    clock = self._game_clock.state
    return build_snapshot(
        view=self,
        clock_state=clock,
        step=self.current_turn,
        max_steps=self.max_turns,
        player_action=self._last_player_action,
        opponent_action=self._last_opponent_action,
        player_combat_results=self._last_player_shooting_results,
        opponent_combat_results=self._last_opponent_shooting_results,
        reward_total=self.last_reward,
        reward_breakdown=self.last_reward_breakdown,
        reward_phase_name=self.phase_manager.current_phase_name,
        reward_phase_index=self.phase_manager.current_phase_index,
        is_terminated=...,  # from last step context
        is_truncated=False,
    )
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `ConfigDict(arbitrary_types_allowed=True)` for numpy | Native Python types in Pydantic models | Pydantic v2 | `model_json_schema()` only works with JSON-native types |
| `model_dump()` + `json.dumps()` | `model_dump_json()` directly | Pydantic v2 | Faster, handles nested models and enums automatically |
| Manual schema writing | `model_json_schema()` | Pydantic v2 | Auto-generated JSON Schema from type annotations |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `schema_version` as a simple string field ("1.0") is sufficient versioning for Phase 1 | Code Examples | Low — can migrate to SemVer or integer later without breaking consumers |
| A2 | Objective control can be recomputed cheaply in `to_snapshot()` | Pitfalls §5 | Low — numpy vectorised ops on 4-8 models × 3 objectives is sub-microsecond |
| A3 | `PairedShootingResult` dataclass in domain is appropriate for the shooting return type change | Code Examples | Medium — could be a named tuple or placed in `state/` instead of domain; planner should decide |

## Open Questions

1. **Where should `PairedShootingResult` live?**
   - What we know: It extends `ShootingResult` with `attacker_idx` and `target_idx`. The data is computed in the facade (`_resolve_shooting_action`), not in the domain's `resolve_shooting()`.
   - What's unclear: Should it live in `domain/shooting.py` (alongside `ShootingResult`), `envs/wargame.py` (local to the facade), or `state/snapshot.py` (with the snapshot models)?
   - Recommendation: Place it in `domain/shooting.py` as a peer of `ShootingResult`. It's still a shooting-resolution value object, just with context. This keeps the domain pure and avoids the facade importing from `state/`.

2. **Should `to_snapshot()` accept a `DistanceCache` parameter?**
   - What we know: Objective control requires distance computation. The env already computes a `DistanceCache` in `step()` but it's player-only. VP scoring computes a separate opponent cache on demand.
   - What's unclear: Should `to_snapshot()` recompute from scratch (simple but redundant), accept the existing player cache (efficient but couples to step internals), or compute only when objective control is requested?
   - Recommendation: Recompute from scratch. The cost is negligible (sub-microsecond for typical scenarios) and keeps `to_snapshot()` self-contained and callable at any point (after reset, mid-step, on demand).

3. **Should the snapshot include weapon profiles per model?**
   - What we know: Weapon stats (attacks, BS, strength, AP, damage, range) are in `ModelConfig` and exposed in `WargameModelObservation`. An LLM evaluator needs them to interpret combat.
   - What's unclear: Include in `ModelSnapshot` (larger model, more fields) or reference by config (smaller, but consumer needs config + snapshot)?
   - Recommendation: Include essential weapon stats in `ModelSnapshot`. An LLM evaluator reading the JSON should have all context in one document without needing the config file.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (installed, configured) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `just test` |
| Full suite command | `just validate` |

### Phase Requirements → Test Map

| Req ID | Behaviour | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SGS-01 | `to_snapshot()` returns `GameStateSnapshot` with all expected fields | integration | `uv run pytest tests/test_snapshot.py::test_to_snapshot_has_all_fields -x` | ❌ Wave 0 |
| SGS-04 | `model_dump_json()` produces valid JSON; `model_json_schema()` produces JSON Schema | unit | `uv run pytest tests/test_snapshot.py::test_json_serialisation -x` | ❌ Wave 0 |
| SGS-11 | Combat results have attacker-target pairing | integration | `uv run pytest tests/test_snapshot.py::test_combat_results_have_pairing -x` | ❌ Wave 0 |
| SGS-12 | Opponent actions recorded before application | integration | `uv run pytest tests/test_snapshot.py::test_opponent_actions_recorded -x` | ❌ Wave 0 |
| SGS-14 | Reward breakdown and phase name in snapshot | integration | `uv run pytest tests/test_snapshot.py::test_reward_breakdown_in_snapshot -x` | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** `uv run pytest tests/test_snapshot.py -x`
- **Per wave merge:** `just test`
- **Phase gate:** `just validate`

### Wave 0 Gaps

- [ ] `tests/test_snapshot.py` — covers SGS-01, SGS-04, SGS-11, SGS-12, SGS-14
- [ ] `tests/conftest.py` — may need a fixture for env-with-opponents-and-shooting (already has `env` but without opponents)

## Sources

### Primary (HIGH confidence)

- Direct codebase analysis — all findings verified against source code:
  - `wargame.py` (env lifecycle, reset/step, shooting resolution, opponent action application)
  - `domain/battle.py`, `domain/entities.py`, `domain/game_clock.py`, `domain/value_objects.py` (state inventory)
  - `domain/shooting.py` (`ShootingResult`, `resolve_shooting()`, `wound_roll_threshold()`)
  - `domain/battle_view.py` (`BattleView` protocol — all properties available)
  - `env_components/actions.py` (`ActionHandler`, `ActionRegistry`, `ActionSlice`, `_decode_action()`)
  - `env_components/distance_cache.py` (`compute_distances()`, `objective_ownership_from_norms_offset()`)
  - `mission/vp_calculator.py` (`DefaultVPCalculator.compute_vp()` — objective control pattern)
  - `reward/phase_manager.py` (`RewardPhaseManager`, `last_reward_breakdown`, `current_phase_name`)
  - `reward/step_context.py` (`StepContext` fields)
  - `types/config.py` (`WargameEnvConfig`, `ModelConfig`, `WeaponProfile`)
  - `types/env_info.py` (`WargameEnvInfo` — existing Pydantic info model)
  - `types/env_observation.py`, `types/model_observation.py` (observation structure)
  - `types/game_timing.py` (`GameState`, `BattlePhase`, `GamePhase`, `PlayerSide` enums)
  - `docs/ddd-envs.md` (architecture guide)
- Local testing of Pydantic v2.11.10 numpy behaviour

### Secondary (MEDIUM confidence)

- `.planning/research/v9-SUMMARY.md`, `v9-STATE-REPRESENTATION.md`, `v9-ARCHITECTURE.md`, `v9-LLM-REPRESENTATION.md` — upstream research findings verified against codebase

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — Pydantic v2 already in use, capabilities verified locally
- Architecture: HIGH — DDD layers, BattleView protocol, registry pattern all well-established; new layer fits naturally
- Pitfalls: HIGH — numpy/Pydantic interaction verified by testing; combat pairing and opponent recording gaps verified by code reading
- Field inventory: HIGH — every field traced to its source in the codebase

**Research date:** 2026-05-23
**Valid until:** 2026-06-23 (stable domain; no external dependencies)
