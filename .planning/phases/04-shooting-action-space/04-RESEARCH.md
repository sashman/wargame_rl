# Phase 4: Shooting Action Space - Research

**Researched:** 2026-04-05
**Domain:** Gymnasium action space extension, phase-gated action masking, target validity
**Confidence:** HIGH

## Summary

Phase 4 extends the flat action space with a `"shooting"` slice in `ActionRegistry`, phase-gated to `BattlePhase.shooting`. The existing infrastructure is well-suited for this: `ActionRegistry` already supports multi-slice registration with per-phase validity, `get_model_action_masks` already handles dead-model masking, and `GameClock`/`turn_execution` already supports multiple player steps per game turn when phases aren't skipped.

The primary engineering work is (a) registering the shooting slice, (b) computing per-model shooting masks that combine phase validity with LOS, range, and target alive status, (c) adding stub `WeaponProfile` with `range` to config, and (d) making `ActionHandler.apply` phase-aware so shooting-phase actions are dispatched differently from movement-phase actions (no-op in Phase 4, resolution in Phase 5).

**Primary recommendation:** Extend `ActionHandler` in place with a `"shooting"` slice and phase-aware `apply()`. Add shooting mask computation as a standalone function that takes positions, alive masks, LOS checker, and weapon ranges — keeping it pure and testable. Inject the mask into the existing `get_model_action_masks` pipeline via a new override mechanism or by extending the registry's per-model mask generation.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** One `env.step()` per phase. Enabling shooting means the agent receives a separate step during `BattlePhase.shooting` (movement slice masked, shooting slice valid) and a separate step during `BattlePhase.movement` (shooting masked, movement valid). Two agent decisions per game turn when shooting is active.
- **D-02:** Shooting phase activation is explicit YAML opt-in via `skip_phases`. Existing configs that don't mention shooting keep current movement-only behaviour.
- **D-03:** Add a `"shooting"` slice to `ActionRegistry` with `number_of_opponent_models` indices, valid only in `frozenset({BattlePhase.shooting})`. Fixed at init, never resized mid-episode.
- **D-04:** Shooting target index K corresponds to opponent model slot K in the observation. Positional alignment must be maintained as an invariant.
- **D-05:** `STAY_ACTION` (index 0) remains valid in all phases including shooting.
- **D-06:** Introduce `WeaponProfile` (Pydantic model) with a `range` field (integer, grid cells). Other weapon stats deferred to Phase 5.
- **D-07:** Add `weapons: list[WeaponProfile]` to `ModelConfig`. Defaults to empty list for backward compatibility. Models with no weapons have all shooting targets masked out.
- **D-08:** Target is "in range" if any of the model's weapons can reach it (max range across all weapons ≥ distance to target).
- **D-09:** Shooting mask for model M targeting opponent K is the AND of: (a) current phase is `BattlePhase.shooting`, (b) model M is alive, (c) opponent K is alive, (d) `has_line_of_sight` from M to K, (e) distance(M, K) ≤ max weapon range of M.
- **D-10:** Dead player models get `STAY_ACTION` only (existing Phase 2 behaviour, unchanged).
- **D-11:** Stateless dispatch. No `model.shooting_target` domain field in Phase 4.
- **D-12:** `ActionHandler.apply` must be phase-aware: during shooting phase, action int maps to a target index rather than a movement displacement. No-op in Phase 4, resolution in Phase 5.

### Claude's Discretion
- Whether to extend `ActionHandler` with phase-aware dispatch or introduce a separate `ShootingHandler`
- Exact distance metric for range check (Euclidean vs Chebyshev vs Manhattan)
- Whether `WeaponProfile` lives in `types/config.py` or a new `types/weapons.py`
- Internal structure of the shooting mask computation (vectorized numpy vs per-model loop)

### Deferred Ideas (OUT OF SCOPE)
- Multi-weapon targeting (Phase 5+) — sub-steps per weapon
- Pointer-network attention for shooting logits
- Decision/event log infrastructure
- Precomputed attacker × defender probability matrices
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ACT-01 | Each model selects an action type per phase: move (movement), shoot (shooting), or stay (any) | Phase-gating via `ActionSlice.valid_phases` on registry slices; shooting slice valid in `{BattlePhase.shooting}` only, movement in `{BattlePhase.movement}` only, stay in `ALL_BATTLE_PHASES` |
| ACT-02 | Shooting actions registered in ActionRegistry as a new slice with shooting-phase validity | Direct `registry.register("shooting", n_opponent_models, frozenset({BattlePhase.shooting}))` — follows identical pattern to stay/movement slices |
| ACT-03 | Action masks combine phase validity, LOS, range, and model alive status | `get_model_action_masks` provides phase+alive base; extend with per-model shooting mask that ANDs in LOS/range/target-alive checks |
| ACT-04 | Total action space grows to accommodate shooting target indices alongside existing movement actions | `ActionRegistry._offset` auto-grows; `TransformerNetwork.policy_head` auto-sizes from `env._action_handler.n_actions` |
| LOS-03 | LOS results used in action masking so agent cannot select invalid shoot targets | `has_line_of_sight` from `domain/los.py` already exists; called per (attacker, target) pair during mask build |
| SHOT-03 | Shooting is only valid during the shooting phase (phase-gated via action masks) | `ActionSlice.valid_phases = frozenset({BattlePhase.shooting})` — registry produces False for all shooting indices in non-shooting phases |
</phase_requirements>

## Project Constraints (from .cursor/rules/)

- **DDD structure**: Domain stays pure (no Gym imports). New action logic goes in `env_components/`. Weapon config goes in `types/config.py`. BattleView may need extension if shooting mask builder needs opponent positions via the view.
- **Dependency direction**: domain → types only; env_components → domain + types; reward/renders → BattleView.
- **Backward compatibility**: New config fields must default to no-op values. Existing YAML configs must work without modification.
- **Testing**: Use Arrange-Act-Assert structure. Prefer real dependencies over mocks. Parameterize where possible. Fixed seeds for determinism.
- **Type hints**: All public functions fully typed. Prefer `list[X]` over `typing.List[X]`.
- **Tool runner**: Use `just` commands. `just format && just lint` for quick iteration.
- **Registry pattern**: Use string identifiers for YAML-configurable subsystems.
- **Config defaults**: Push defaults into Pydantic `Field(default=...)`, avoid magic numbers in factories.

## Standard Stack

### Core (already in project)

| Library | Version | Purpose | Status |
|---------|---------|---------|--------|
| Gymnasium | 1.x | Action space `Discrete(n)`, observation spaces | In use |
| NumPy | current | Vectorized mask computation, distance arrays | In use |
| Pydantic | v2 | `WeaponProfile` model, `ModelConfig` extension | In use |
| PyTorch | current | Network output shape adapts via `n_actions` | In use |

### No new dependencies required

Phase 4 is purely an extension of existing infrastructure. No external libraries needed.

## Architecture Patterns

### Recommended Approach: Extend ActionHandler

**What:** Add shooting slice registration and phase-aware `apply()` to the existing `ActionHandler` class rather than creating a separate handler.

**Why:**
1. `ActionHandler` already owns the `ActionRegistry` and is the single point for action space construction
2. A separate `ShootingHandler` would split registry ownership (movement in one, shooting in another) — the registry's contiguous offset model works best with a single registrant
3. Phase-awareness in `apply()` is a small addition (~10 lines): check if the action falls in the shooting slice; if so, skip displacement computation (Phase 4 no-op, Phase 5 resolution)
4. The env already uses `self._action_handler` everywhere — no wiring changes needed

**Pattern:**

```python
# In ActionHandler.__init__:
if config.number_of_opponent_models > 0:
    self._shooting_slice = self._registry.register(
        "shooting",
        config.number_of_opponent_models,
        frozenset({BattlePhase.shooting}),
    )
else:
    self._shooting_slice = None
```

```python
# In ActionHandler.apply — phase-aware dispatch:
def apply(self, action, wargame_models, board_width, board_height,
          action_space, *, phase: BattlePhase = BattlePhase.movement):
    for i, act in enumerate(action.actions):
        model = wargame_models[i]
        if not model.is_alive:
            continue
        if act == STAY_ACTION:
            continue
        if self._shooting_slice and self._shooting_slice.start <= act < self._shooting_slice.end:
            # Shooting action — no-op in Phase 4; Phase 5 adds resolution here
            continue
        # Existing movement logic
        model.previous_location = model.location.copy()
        displacement = self._decode_action(act)
        model.location = np.clip(...)
```

### Shooting Mask as Standalone Function

**What:** A pure function `compute_shooting_masks(...)` that takes positions, alive masks, LOS checker, and weapon ranges; returns per-model shooting mask arrays.

**Why:**
1. Pure functions are testable without env instantiation
2. Separates the mask computation logic from the registry infrastructure
3. Can be vectorized with numpy for performance
4. The `build_observation` pipeline calls it and overlays the result onto the registry-generated base mask

**Pattern:**

```python
def compute_shooting_masks(
    player_positions: np.ndarray,   # (n_player, 2)
    opponent_positions: np.ndarray, # (n_opponent, 2)
    player_alive: np.ndarray,       # (n_player,) bool
    opponent_alive: np.ndarray,     # (n_opponent,) bool
    player_max_ranges: np.ndarray,  # (n_player,) float — max weapon range per model
    has_los_fn: Callable[[int, int, int, int], bool],
) -> np.ndarray:
    """Return (n_player, n_opponent) bool mask — True where model can shoot target."""
    n_player, n_opponent = len(player_positions), len(opponent_positions)
    mask = np.zeros((n_player, n_opponent), dtype=bool)

    for m in range(n_player):
        if not player_alive[m] or player_max_ranges[m] <= 0:
            continue
        mx, my = int(player_positions[m, 0]), int(player_positions[m, 1])
        for k in range(n_opponent):
            if not opponent_alive[k]:
                continue
            kx, ky = int(opponent_positions[k, 0]), int(opponent_positions[k, 1])
            dist = np.sqrt((mx - kx)**2 + (my - ky)**2)
            if dist > player_max_ranges[m]:
                continue
            if has_los_fn(mx, my, kx, ky):
                mask[m, k] = True
    return mask
```

### Mask Integration Point

The per-model shooting mask must be overlaid onto the registry's phase-gated base mask. The current `get_model_action_masks` returns `(n_models, n_actions)` with the shooting slice either all-True (shooting phase) or all-False (other phases). The overlay zeros out specific shooting targets per model.

**Two approaches:**

1. **Post-process in observation builder** — after `get_model_action_masks`, apply `compute_shooting_masks` to the shooting slice. This keeps the registry simple and pure.

2. **Extend `get_model_action_masks` with a callback** — pass a `per_model_mask_fn` that the registry calls for custom slices.

**Recommendation: Approach 1 (post-process in observation builder).** Keeps the registry general-purpose and puts the shooting-specific logic in a clear, testable location. The observation builder already receives `view: BattleView` (which has model positions, opponent models, and LOS) and `action_registry`.

```python
# In observation_builder.py, inside build_observation:
if action_registry is not None:
    phase = view.game_clock_state.phase or BattlePhase.movement
    player_alive = alive_mask_for(view.player_models)
    action_mask = action_registry.get_model_action_masks(
        phase, len(view.player_models), alive_mask=player_alive
    )
    # Overlay shooting validity if shooting slice exists
    shooting_slice = action_registry.slice_for("shooting") if "shooting" in ... else None
    if shooting_slice is not None and phase == BattlePhase.shooting:
        shooting_validity = compute_shooting_masks(...)
        action_mask[:, shooting_slice.start:shooting_slice.end] &= shooting_validity
```

### WeaponProfile Placement

**Recommendation: In `types/config.py` alongside `ModelConfig`.**

`WeaponProfile` is a Pydantic configuration model — it describes what weapons a model has, not runtime state. It belongs with other config types. It's small (one field in Phase 4) and tightly coupled to `ModelConfig.weapons`. A separate `types/weapons.py` is premature given the single-field model.

```python
class WeaponProfile(BaseModel):
    """Weapon configuration for a model."""
    range: int = Field(gt=0, description="Maximum range in grid cells")
    # Phase 5 adds: attacks, ballistic_skill, strength, ap, damage
```

### Distance Metric

**Recommendation: Euclidean (L2 norm).** The existing `DistanceCache` uses `np.linalg.norm(..., ord=2)` (Euclidean) for all distance computations. Using the same metric for shooting range checks maintains consistency and avoids confusing mismatches (e.g. a model at diagonal distance 7.07 being "in range" for movement but "out of range" for shooting if different metrics were used).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Phase-aware action masking | Custom mask manager | `ActionRegistry` with `valid_phases` | Already handles multi-slice phase gating; adding a slice is one line |
| Dead model exclusion | Manual per-step checks | `get_model_action_masks(alive_mask=...)` | Already restricts dead models to STAY_ACTION |
| LOS checking | New ray-trace implementation | `has_line_of_sight` from `domain/los.py` | Bresenham already implemented and tested in Phase 3 |
| Distance computation | Per-pair manual calculation | NumPy vectorized broadcasting | `player_locs[:, None, :] - opp_locs[None, :, :]` gives all pairwise deltas in one operation |
| Network output resizing | Manual dimension tracking | `TransformerNetwork.from_env()` reads `n_actions` from handler | Auto-adapts when shooting slice is registered |

## Common Pitfalls

### Pitfall 1: Observation-Action Index Misalignment
**What goes wrong:** Target index K in the shooting action slice doesn't correspond to opponent K in the observation tensor.
**Why it happens:** If opponent model ordering in observations or actions is sorted/shuffled independently, the positional correspondence breaks.
**How to avoid:** Ensure both `build_observation` (which builds `opponent_models` list) and the shooting slice use the same underlying list order (`view.opponent_models`). Never sort or reorder either independently. Add an assertion or test that validates the correspondence.
**Warning signs:** Agent learns to always select the same target index regardless of opponent positions; training loss plateaus.

### Pitfall 2: Mask All-False During Shooting Phase
**What goes wrong:** A model in the shooting phase has all shooting targets masked out AND stay is also masked, leaving zero valid actions.
**Why it happens:** Bug in the mask overlay where stay gets cleared during the shooting-specific masking.
**How to avoid:** The shooting mask overlay must only modify indices `shooting_slice.start:shooting_slice.end`. STAY_ACTION (index 0) is in the "stay" slice and must never be touched by shooting mask logic. Always assert `mask[m, STAY_ACTION] == True` after mask computation for alive models.
**Warning signs:** `ValueError` during action sampling; NaN in policy logits from `-inf` everywhere.

### Pitfall 3: Backward Incompatibility with Existing Configs
**What goes wrong:** Existing YAML configs (no opponents, no weapons) break or change behavior after the code change.
**Why it happens:** `WeaponProfile`/`weapons` field doesn't default properly; shooting slice registration changes `n_actions` when it shouldn't; `skip_phases` default changes.
**How to avoid:**
- `ModelConfig.weapons` defaults to `[]` (no weapons → all shooting masked)
- Shooting slice only registered when `number_of_opponent_models > 0`
- `skip_phases` default remains `NON_MOVEMENT_PHASES` (includes `BattlePhase.shooting`)
- Existing configs never see shooting phase because it's in `skip_phases` by default
**Warning signs:** Existing tests fail; network checkpoint loading errors (shape mismatch).

### Pitfall 4: ActionHandler.apply Not Receiving Phase
**What goes wrong:** `apply()` doesn't know the current phase, so it tries to decode a shooting action as a movement displacement, causing out-of-bounds array access.
**Why it happens:** Current `apply()` signature doesn't accept `phase`. The caller (`_apply_player_action` in wargame.py) doesn't pass it.
**How to avoid:** Add `phase: BattlePhase` parameter to `apply()`. Have `_apply_player_action` pass `self._game_clock.state.phase`. Check if the action int falls in the shooting slice — if so, handle appropriately (no-op in Phase 4).
**Warning signs:** `IndexError` in `_decode_action` when action > max movement index; numpy array index out of bounds.

### Pitfall 5: LOS Check Performance on Large Boards
**What goes wrong:** `has_line_of_sight` is called for every (model, opponent) pair every step, causing performance regression.
**Why it happens:** Bresenham trace is O(board_diagonal) per call; with N models × M opponents, total cost is O(N×M×diagonal).
**How to avoid:** For Phase 4 scale (4-12 models), the O(N×M×diagonal) cost is negligible. If scale grows, consider caching LOS results per step (positions only change during movement phase) or vectorizing the trace. No action needed in Phase 4.
**Warning signs:** Training steps/second drops significantly when shooting is enabled.

### Pitfall 6: Opponent ActionHandler Not Getting Shooting Slice
**What goes wrong:** The opponent `ActionHandler` (created separately in `WargameEnv.__init__`) doesn't register the shooting slice, so opponent policies can't select shooting targets.
**Why it happens:** Opponent handler is constructed identically to player handler but may not have the right context for number of *player* models (opponents shoot at player models, not at other opponents).
**How to avoid:** This is a **Phase 5** concern (opponents don't shoot in Phase 4 since there's no resolution). However, design the shooting slice registration to be parameterizable: `ActionHandler.__init__` should accept `n_shoot_targets` (defaulting to `number_of_opponent_models` for the player handler, `number_of_wargame_models` for the opponent handler).
**Warning signs:** Opponent policy crashes when shooting phase is active.

## Code Examples

### ActionRegistry Shooting Slice Registration

```python
# Follows the exact pattern of stay and movement slices (actions.py lines 155-161)
# Source: existing ActionHandler.__init__

self._registry.register("stay", 1, ALL_BATTLE_PHASES)
self._registry.register(
    "movement",
    self._n_move_actions,
    frozenset({BattlePhase.movement}),
)

# NEW: register shooting slice (only when opponents exist)
n_shoot_targets = n_shoot_targets or 0
if n_shoot_targets > 0:
    self._shooting_slice = self._registry.register(
        "shooting",
        n_shoot_targets,
        frozenset({BattlePhase.shooting}),
    )
else:
    self._shooting_slice = None
```

### Flat Action Space Layout (after Phase 4)

```
Index:  [0]       [1 .. N]      [N+1 .. N+M]
Slice:  [stay]    [movement]    [shooting]
Valid:  all       movement      shooting
        phases    phase only    phase only

Where:
  N = n_movement_angles × n_speed_bins  (e.g. 96)
  M = number_of_opponent_models         (e.g. 4)
  Total actions = 1 + N + M             (e.g. 101)
```

### WeaponProfile Config

```python
class WeaponProfile(BaseModel):
    """Weapon stat block. Phase 4 uses only `range`; Phase 5 adds resolution stats."""
    range: int = Field(gt=0, description="Maximum range in grid cells")

class ModelConfig(BaseModel):
    # ... existing fields ...
    weapons: list[WeaponProfile] = Field(
        default_factory=list,
        description="Weapon profiles for this model. Empty = cannot shoot.",
    )
```

### YAML Example (Shooting-Enabled Config)

```yaml
config_name: 4v4_with_shooting
number_of_wargame_models: 4
number_of_opponent_models: 4
board_width: 60
board_height: 44

# Enable shooting phase for the agent
skip_phases: [command, charge, fight]  # omit 'shooting' to enable it

models:
  - { group_id: 0, weapons: [{ range: 24 }] }
  - { group_id: 0, weapons: [{ range: 24 }] }
  - { group_id: 1, weapons: [{ range: 12 }, { range: 48 }] }
  - { group_id: 1, weapons: [{ range: 12 }] }

opponent_models:
  - { group_id: 0 }
  - { group_id: 0 }
  - { group_id: 1 }
  - { group_id: 1 }

opponent_policy:
  type: scripted_advance_to_objective
```

### Per-Model Shooting Mask Computation

```python
def compute_shooting_masks(
    player_positions: np.ndarray,
    opponent_positions: np.ndarray,
    player_alive: np.ndarray,
    opponent_alive: np.ndarray,
    player_max_ranges: np.ndarray,
    has_los_fn: Callable[[int, int, int, int], bool],
) -> np.ndarray:
    """(n_player, n_opponent) bool mask — True where model M can shoot target K."""
    n_player = len(player_positions)
    n_opponent = len(opponent_positions)
    mask = np.zeros((n_player, n_opponent), dtype=bool)

    if n_opponent == 0:
        return mask

    # Vectorized distance computation
    deltas = player_positions[:, np.newaxis, :] - opponent_positions[np.newaxis, :, :]
    distances = np.linalg.norm(deltas, axis=2)  # (n_player, n_opponent)

    for m in range(n_player):
        if not player_alive[m] or player_max_ranges[m] <= 0:
            continue
        mx, my = int(player_positions[m, 0]), int(player_positions[m, 1])
        for k in range(n_opponent):
            if not opponent_alive[k]:
                continue
            if distances[m, k] > player_max_ranges[m]:
                continue
            kx, ky = int(opponent_positions[k, 0]), int(opponent_positions[k, 1])
            if has_los_fn(mx, my, kx, ky):
                mask[m, k] = True
    return mask
```

### Mask Integration in Observation Builder

```python
# After base mask is built from registry (phase + alive gating):
if shooting_slice is not None:
    if phase == BattlePhase.shooting:
        shooting_validity = compute_shooting_masks(
            player_positions=...,
            opponent_positions=...,
            player_alive=player_alive,
            opponent_alive=alive_mask_for(view.opponent_models),
            player_max_ranges=_max_weapon_ranges(view.config, len(view.player_models)),
            has_los_fn=view.has_line_of_sight_between_cells,
        )
        # Overlay: AND with base mask's shooting region
        action_mask[:, shooting_slice.start:shooting_slice.end] &= shooting_validity
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single-phase env (movement only) | Multi-phase potential via GameClock + skip_phases | Phase architecture already exists | Shooting phase is an activation of existing infrastructure, not new infra |
| Flat action space, all movement | Sliced registry with phase-gating | Phase 2 (alive masking) | Shooting slice follows identical registration pattern |
| No opponent models | Opponent models with scripted policies | Prior milestone | Opponents provide the shoot targets |

## Open Questions

1. **Opponent shooting in Phase 4?**
   - What we know: D-11/D-12 say stateless dispatch, no-op in Phase 4. Opponent policies currently select movement actions only.
   - What's unclear: Should opponent `ActionHandler` also register a shooting slice for symmetry (even though opponent policies don't use it yet)?
   - Recommendation: Register the slice on the opponent handler too (with `n_shoot_targets = number_of_wargame_models`). This prevents shape mismatches when opponent shooting is enabled in Phase 5. The scripted policies will just select STAY during shooting phase since they don't know about targets — acceptable behavior.

2. **max_turns_override interaction with two steps per turn**
   - What we know: With shooting enabled, each game turn produces two agent steps (movement + shooting). The `max_turns_override=100` counts agent steps, not game turns.
   - What's unclear: Should this value be adjusted for shooting-enabled configs?
   - Recommendation: Document this interaction. Users should set `max_turns_override` to `2 × desired_game_turns` when shooting is enabled. This is a config-level concern, not a code change.

3. **BattleView extension for shooting masks**
   - What we know: The observation builder needs opponent positions, LOS, and weapon ranges. `BattleView` exposes `opponent_models` and the env has `has_line_of_sight_between_cells`.
   - What's unclear: The observation builder receives `view: BattleView` but `has_line_of_sight_between_cells` is on the env, not on `BattleView`.
   - Recommendation: Either add `has_line_of_sight_between_cells` to `BattleView` protocol, or pass the LOS function separately to the mask builder. The cleaner approach is adding it to `BattleView` since it's a read-only query.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (latest via UV) |
| Config file | `pyproject.toml` [tool.pytest.ini_options] |
| Quick run command | `just test` |
| Full suite command | `just validate` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ACT-01 | Each model selects action type per phase | integration | `uv run pytest tests/test_shooting_action.py::test_movement_and_shooting_alternate -x` | ❌ Wave 0 |
| ACT-02 | Shooting slice registered in ActionRegistry | unit | `uv run pytest tests/test_shooting_action.py::test_shooting_slice_registered -x` | ❌ Wave 0 |
| ACT-03 | Action masks combine phase, LOS, range, alive | unit + integration | `uv run pytest tests/test_shooting_action.py::test_shooting_mask_combines_conditions -x` | ❌ Wave 0 |
| ACT-04 | Action space grows for shooting targets | unit | `uv run pytest tests/test_shooting_action.py::test_action_space_includes_shooting -x` | ❌ Wave 0 |
| LOS-03 | LOS used in shooting mask | unit | `uv run pytest tests/test_shooting_action.py::test_los_blocks_shooting_mask -x` | ❌ Wave 0 |
| SHOT-03 | Shooting only valid in shooting phase | unit | `uv run pytest tests/test_shooting_action.py::test_shooting_masked_in_movement_phase -x` | ❌ Wave 0 |

### Additional Critical Tests
| Behavior | Test Type | File Exists? |
|----------|-----------|-------------|
| Backward compat: existing configs unchanged `n_actions` | regression | ❌ Wave 0 |
| Backward compat: no-opponent env has no shooting slice | regression | ❌ Wave 0 |
| Dead model gets STAY only during shooting phase | unit | ❌ Wave 0 (extension of existing test_wounds pattern) |
| Models with no weapons have all shooting masked | unit | ❌ Wave 0 |
| Tensor pipeline handles enlarged action mask | integration | ❌ Wave 0 (extension of existing test_action_masking pattern) |
| Transformer network auto-sizes with shooting | integration | ❌ Wave 0 |
| STAY always valid in shooting phase | unit | ❌ Wave 0 |
| Out-of-range target masked | unit | ❌ Wave 0 |
| ActionHandler.apply no-ops on shooting actions | unit | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `just test`
- **Per wave merge:** `just validate`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/test_shooting_action.py` — covers all ACT-*, LOS-03, SHOT-03 requirements
- [ ] Test fixtures for shooting-enabled env configs with fixed positions and LOS scenarios

## Sources

### Primary (HIGH confidence)
- **Codebase analysis** — `actions.py`, `observation_builder.py`, `wargame.py`, `game_clock.py`, `los.py`, `entities.py`, `config.py`, `net.py`, `types/` — full read of all implementation touchpoints
- **Prior phase contexts** — Phases 1-3 CONTEXT.md files (referenced in canonical refs)
- **Architecture docs** — `docs/ddd-envs.md`, `docs/tabletop-rules-reference.md`
- **Test suite** — `test_action_masking.py`, `test_wounds.py`, `test_los.py`, `conftest.py` for existing patterns

### Secondary (MEDIUM confidence)
- **Gymnasium docs** — `spaces.Discrete`, `spaces.Tuple` behavior with dynamically-sized action spaces
- **Tabletop rules reference** — Shooting phase rules, eligible targets, range, LOS requirements

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new deps; all infrastructure exists
- Architecture: HIGH — patterns are clear from codebase analysis; extension points are well-documented in ddd-envs.md
- Pitfalls: HIGH — identified from code structure analysis; misalignment and backward compat risks are concrete and testable

**Research date:** 2026-04-05
**Valid until:** 2026-05-05 (stable; no external deps to go stale)
