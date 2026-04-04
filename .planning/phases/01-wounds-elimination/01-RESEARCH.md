# Phase 1: Wounds & Elimination - Research

**Researched:** 2026-04-02
**Domain:** Domain entity state, alive-filtering across environment loop, termination conditions
**Confidence:** HIGH

## Summary

Phase 1 adds durable wound state to `WargameModel`, introduces flag-based elimination (`is_alive` derived from `current_wounds > 0`), and extends termination logic to end episodes when all models on a side are eliminated. No damage source is introduced — wounds are exercised via unit tests calling `take_damage` directly. Shooting (Phase 5) will be the first real consumer.

The codebase already has the data scaffolding in place: `WargameModel.stats` stores `max_wounds` and `current_wounds`, `reset_for_episode()` resets wounds, and `ModelConfig.max_wounds` is configurable. The primary work is (1) adding mutation + property methods on `WargameModel`, (2) threading alive-filtering through every model iteration site, and (3) extending termination with an `all_eliminated` condition.

**Primary recommendation:** Add `take_damage(amount)` and `is_alive` property to `WargameModel`, then systematically guard every model iteration loop on `is_alive` — action application, distance computation, reward calculation, objective ownership, opponent policy, and rendering.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Add a `take_damage(amount)` method on `WargameModel` that reduces `current_wounds` (clamped to 0). This is the sole entry point for wound reduction across the codebase.
- **D-02:** No in-env damage source in Phase 1. Wounds are exercised via unit tests that call `take_damage` directly on domain objects during step sequences. Shooting resolution (Phase 5) will be the first real caller.
- **D-03:** Flag-based elimination. Models stay in `player_models` / `opponent_models` lists at their original index. An `is_alive` property on `WargameModel` returns `current_wounds > 0`.
- **D-04:** Eliminated models are excluded from action selection, movement application, and objective control checks by filtering on `is_alive`. Array shapes never change mid-episode.
- **D-05:** All iteration over models that applies actions, checks OC, or computes distances must guard on `is_alive`. This includes `ActionHandler`, distance cache, and any reward calculators that inspect model positions.
- **D-06:** Change `ModelConfig.max_wounds` default from 100 to 1 (standard tabletop infantry). Safe because no damage source exists until Phase 5; configs designed for combat will set explicit values.
- **D-07:** Existing YAML configs without `max_wounds` will get 1-wound models. This is acceptable — combat-era configs will be purpose-built with explicit wound values.
- **D-08:** Extend `is_battle_over` in `domain/termination.py` with an `all_eliminated` condition: episode ends when all player models or all opponent models are eliminated.
- **D-09:** This condition composes with existing termination (max turns, clock, all-at-objectives) — any condition being true ends the episode.

### Claude's Discretion
- Whether `take_damage` clamps at 0 or raises on negative input
- Internal implementation of alive-filtering helpers (property on Battle vs utility function)
- Test structure and parameterization for wound/elimination scenarios

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| WOUND-01 | Each model has configurable max wounds and tracks current wounds during an episode | `WargameModel.stats` already stores both values; `ModelConfig.max_wounds` is configurable. Add `take_damage()` for mutation and `is_alive` property for state query. |
| WOUND-02 | Models reduced to 0 wounds are eliminated and removed from active play | Flag-based elimination via `is_alive` property (`current_wounds > 0`). Models remain in lists but are excluded from all gameplay logic. |
| WOUND-03 | Eliminated models are excluded from action selection, movement, and objective control | Requires alive-guards in `ActionHandler.apply`, `compute_distances`, `objective_ownership_from_norms_offset`, reward calculators, opponent policies, and renderer. |
| WOUND-05 | Episode terminates when all models on one side are eliminated | Extend `is_battle_over` with `all_eliminated` flag, computed in `WargameEnv.step()` by checking `is_alive` on all player/opponent models. |
</phase_requirements>

## Project Constraints (from .cursor/rules/)

- Follow DDD structure: domain logic in `domain/`, read-only `BattleView` for reward/renders, dependency direction domain → types only
- Domain does not import from `env_components`, `reward`, or `renders`
- Keep new config fields optional/defaulted for backward compatibility with existing YAML
- Use `just validate` (format + lint + test) before pushing
- Type hints on all public functions; strict mypy
- Ruff formatting (88 cols, double quotes)
- Prefer simplicity over cleverness
- Tests: Arrange–Act–Assert, deterministic (fixed seeds), parameterized

## Standard Stack

No new libraries are needed. This phase is purely internal domain changes.

### Core (existing, unchanged)
| Library | Version | Purpose | Note |
|---------|---------|---------|------|
| gymnasium | >=1.0.0,<2.0.0 | RL environment | No changes to Gym interface |
| numpy | latest | Array operations | Distance computations, alive masking |
| pydantic | latest | Config models | `ModelConfig.max_wounds` default change |
| pytest | latest | Testing | New wound/elimination tests |

## Architecture Patterns

### Recommended Approach: Flag-Based Elimination

The locked decision (D-03) specifies flag-based elimination: models stay in their list at the same index, and an `is_alive` property determines whether they participate in gameplay.

**Why this is correct for RL:**
- Fixed-size observation/action spaces — Gymnasium environments must declare space shapes at init time. Removing models mid-episode would require variable-length spaces or padding, which complicates the tensor pipeline.
- Index stability — reward calculators, observation builders, and distance caches all use `model_idx` indexing. Keeping models in place preserves index correspondence.
- Simplicity — a boolean guard is cheaper than managing dynamic lists and re-indexing.

### Pattern: Alive Guard on Model Iteration

Every loop that iterates models and applies gameplay effects must guard on `is_alive`. The canonical pattern:

```python
for i, model in enumerate(models):
    if not model.is_alive:
        continue
    # ... gameplay logic ...
```

### Touch-Point Inventory

All sites that iterate over models and need alive-guards:

| File | Function/Method | What It Does | Guard Needed |
|------|----------------|--------------|--------------|
| `domain/entities.py` | `WargameModel` | Entity definition | Add `take_damage()`, `is_alive` property |
| `domain/termination.py` | `is_battle_over` | Termination check | Add `all_eliminated` parameter |
| `env_components/actions.py` | `ActionHandler.apply` | Applies movement to models | Skip dead models |
| `env_components/distance_cache.py` | `compute_distances` | Computes model-objective distances | Handle dead models (see below) |
| `env_components/distance_cache.py` | `DistanceCache.all_models_at_objectives` | Checks all models at objectives | Only check alive models |
| `env_components/distance_cache.py` | `objective_ownership_from_norms_offset` | Determines objective control | Only count alive models |
| `env_components/distance_cache.py` | `DistanceCache.min_distances_to_same_group` | Group cohesion distances | Only consider alive models |
| `env_components/observation_builder.py` | `build_observation`, `build_info` | Builds obs/info | Keep dead models in arrays (Phase 2 adds alive flags); no guard needed in Phase 1 |
| `reward/phase_manager.py` | `RewardPhaseManager.calculate_reward` | Per-model reward averaging | Only average over alive models |
| `reward/calculators/closest_objective.py` | `ClosestObjectiveCalculator.calculate` | Per-model reward | Caller (phase_manager) skips dead; calculator receives only alive models |
| `reward/calculators/group_cohesion.py` | `GroupCohesionCalculator.calculate` | Per-model group penalty | Same — caller skips dead |
| `mission/vp_calculator.py` | `DefaultVPCalculator.compute_vp` | VP scoring from objective control | Uses distance caches that need alive-filtering |
| `opponent/scripted_advance_to_objective_policy.py` | `select_action` | Opponent movement | Skip dead opponent models (STAY_ACTION for dead) |
| `renders/human.py` | `_draw_agent`, `_draw_opponent_models` | Rendering | Visually distinguish dead models |
| `wargame.py` | `WargameEnv.step` | Main env loop | Compute `all_eliminated` flag, pass to `is_battle_over` |

### Distance Cache Strategy for Dead Models

Two viable approaches:

**Option A — Filter at computation time:** `compute_distances` only computes for alive models, using sentinel values (e.g. `inf` distances) for dead ones. This keeps downstream consumers simple but requires alive-state input.

**Option B — Filter at consumption time:** `compute_distances` computes for all models (including dead), and consumers filter. Simpler function signature but more guard points.

**Recommendation: Option A** — pass alive status into distance functions and use sentinel values for dead models. This centralizes the logic. Specifically:
- `compute_distances`: accept optional `alive_mask: np.ndarray | None`. When provided, set distance norms for dead models to `inf` so they never count as "at objective" or "controlling objective."
- `objective_ownership_from_norms_offset`: accept optional alive masks for player and opponent. Dead models' distances set to `inf` means they naturally fail the `<= radius` check.
- `all_models_at_objectives`: accept optional alive mask; only check alive models.

### Reward Calculation for Dead Models

The `RewardPhaseManager.calculate_reward` loop currently iterates all `view.player_models` and averages. With elimination:

```python
alive_models = [(i, m) for i, m in enumerate(view.player_models) if m.is_alive]
n_alive = len(alive_models)
# ... calculate per-model reward only for alive models ...
# Average over n_alive (not total n_models)
```

This ensures dead models don't dilute rewards and don't generate spurious reward signals.

### Action Masking for Dead Models

`ActionHandler.apply` receives all model actions. For dead models, the action should be forced to `STAY_ACTION` (no movement). Two places to enforce this:

1. **In `apply()`:** Skip movement for dead models regardless of action value.
2. **In action mask:** Set all actions to False for dead models except STAY_ACTION.

Both should be done for safety (defense in depth), but the action mask is the primary enforcement point — it tells the policy network not to select movement actions for dead models.

The current `ActionRegistry.get_model_action_masks` returns the same mask for all models. To per-model mask dead models, it needs alive status:

```python
def get_model_action_masks(
    self, phase: BattlePhase, n_models: int,
    alive_mask: np.ndarray | None = None,
) -> np.ndarray:
    masks = np.tile(self.get_action_mask(phase), (n_models, 1))
    if alive_mask is not None:
        # Dead models: only STAY_ACTION is valid
        for i in range(n_models):
            if not alive_mask[i]:
                masks[i, :] = False
                masks[i, STAY_ACTION] = True
    return masks
```

### Termination Extension

`is_battle_over` currently takes: `clock`, `current_turn`, `max_turns`, `max_turns_override`, `all_models_at_objectives_flag`.

Add `all_eliminated: bool = False` — episode ends when all player models OR all opponent models are eliminated:

```python
def is_battle_over(
    clock: GameClock,
    current_turn: int,
    max_turns: int,
    max_turns_override: int | None,
    all_models_at_objectives_flag: bool,
    all_eliminated: bool = False,
) -> bool:
    if all_eliminated:
        return True
    # ... existing logic unchanged ...
```

The `all_eliminated` flag is computed in `WargameEnv.step()` by checking:
```python
all_player_eliminated = all(not m.is_alive for m in self.wargame_models)
all_opponent_eliminated = (
    bool(self.opponent_models)
    and all(not m.is_alive for m in self.opponent_models)
)
all_eliminated = all_player_eliminated or all_opponent_eliminated
```

Note: if there are no opponents (`opponent_models` is empty), only player elimination triggers termination.

### Anti-Patterns to Avoid

- **Removing models from lists mid-episode:** Breaks index stability, observation shape, and all `model_idx`-based logic.
- **Changing array shapes when models die:** Gymnasium spaces are declared at init. Shape changes cause crashes.
- **Forgetting to guard a single iteration site:** One unguarded loop means dead models can still move, capture objectives, or earn rewards. The touch-point inventory above must be exhaustive.
- **Averaging rewards over total models instead of alive models:** Dead models contribute 0 reward but inflate the divisor, making rewards artificially small.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Alive filtering on arrays | Custom array slicing per call site | Centralized alive mask in `compute_distances` with sentinel values | Single source of truth, less error-prone |
| Dynamic model removal | Remove/re-add models from lists | Flag-based `is_alive` property | RL environments need fixed shapes |

## Common Pitfalls

### Pitfall 1: Forgetting an Iteration Site
**What goes wrong:** Dead models still participate in some game mechanic (e.g., objective control, group cohesion).
**Why it happens:** Many files iterate models independently. Easy to miss one.
**How to avoid:** Use the touch-point inventory above as a checklist. Search for all `for.*model` loops and `enumerate.*models` patterns.
**Warning signs:** Tests where a dead model still moves, captures an objective, or affects reward.

### Pitfall 2: Reward Denominator Bug
**What goes wrong:** Per-model rewards averaged over `len(player_models)` instead of alive count, making rewards vanish as models die.
**Why it happens:** The current `calculate_reward` uses `n_models = len(view.player_models)` for averaging.
**How to avoid:** Replace with alive count. If all models are dead (n_alive == 0), return 0.0.
**Warning signs:** Reward drops sharply when models are eliminated in tests.

### Pitfall 3: Distance Cache Inconsistency
**What goes wrong:** `compute_distances` computes distances for dead models, and `all_models_at_objectives` treats them as "not at objective," blocking success.
**Why it happens:** Dead models have real positions but shouldn't count.
**How to avoid:** Use `inf` sentinel for dead model distances, or filter in `all_models_at_objectives`.
**Warning signs:** Episode never terminates via "all at objectives" even when all alive models are at objectives.

### Pitfall 4: Opponent Policy Crashes on Dead Models
**What goes wrong:** `ScriptedAdvanceToObjectivePolicy` computes centroid including dead model positions, or tries to compute direction for a dead model.
**Why it happens:** Policy iterates all opponent models.
**How to avoid:** Skip dead models (emit STAY_ACTION). Compute centroid only from alive models.
**Warning signs:** NaN in opponent actions, incorrect centroid pulling toward dead model locations.

### Pitfall 5: Default Change Breaks Existing Configs
**What goes wrong:** Changing `max_wounds` default from 100 to 1 causes model observation space upper bound to no longer cover the value 100.
**Why it happens:** `WargameModel.to_space()` defines `max_wounds` space as `Box(0, 100)`. The default change is fine because the space already allows 1.
**How to avoid:** Verify `to_space()` bounds are still valid for both default=1 and explicit higher values.
**Warning signs:** None expected — space is already [0, 100].

### Pitfall 6: `all_eliminated` Without Opponents
**What goes wrong:** When `number_of_opponent_models=0`, checking "all opponent models eliminated" on an empty list returns True (vacuous truth), immediately ending the episode.
**Why it happens:** `all(not m.is_alive for m in [])` returns `True` in Python.
**How to avoid:** Only check opponent elimination when opponents exist: `bool(opponent_models) and all(not m.is_alive for m in opponent_models)`.
**Warning signs:** Episodes end immediately on reset with 0-opponent configs.

## Code Examples

### Adding `take_damage` and `is_alive` to WargameModel

```python
# In domain/entities.py

@property
def is_alive(self) -> bool:
    """True while the model has wounds remaining."""
    return self.stats["current_wounds"] > 0

def take_damage(self, amount: int) -> None:
    """Reduce current wounds by amount (clamped to 0).

    Sole entry point for wound reduction.
    """
    self.stats["current_wounds"] = max(
        0, self.stats["current_wounds"] - amount
    )
```

### Alive-guarded action application

```python
# In env_components/actions.py, ActionHandler.apply

def apply(self, action, wargame_models, board_width, board_height, action_space):
    for i, act in enumerate(action.actions):
        model = wargame_models[i]
        if not model.is_alive:
            continue  # Dead models don't move
        if not action_space[i].contains(act):
            raise ValueError(...)
        model.previous_location = model.location.copy()
        displacement = self._decode_action(act)
        model.location = np.clip(...)
```

### Alive-filtered reward averaging

```python
# In reward/phase_manager.py

alive_models = [(i, m) for i, m in enumerate(view.player_models) if m.is_alive]
n_alive = len(alive_models)

for i, model in alive_models:
    for name, pm_calc in phase.per_model_calculators:
        per_model_sums[name] += pm_calc.weight * pm_calc.calculate(i, model, view, ctx)

if n_alive > 0:
    for name in per_model_sums:
        per_model_sums[name] /= n_alive
```

### Extended termination

```python
# In domain/termination.py

def is_battle_over(
    clock: GameClock,
    current_turn: int,
    max_turns: int,
    max_turns_override: int | None,
    all_models_at_objectives_flag: bool,
    all_eliminated: bool = False,
) -> bool:
    if all_eliminated:
        return True
    if max_turns_override is not None:
        return current_turn >= max_turns or all_models_at_objectives_flag
    return (
        current_turn >= max_turns
        or clock.is_game_over
        or all_models_at_objectives_flag
    )
```

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (latest, via uv) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `just test` |
| Full suite command | `just validate` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| WOUND-01 | Model starts with max_wounds, take_damage reduces current_wounds | unit | `uv run pytest tests/test_wounds.py::test_wound_tracking -x` | ❌ Wave 0 |
| WOUND-02 | Model at 0 wounds reports `is_alive == False` | unit | `uv run pytest tests/test_wounds.py::test_elimination -x` | ❌ Wave 0 |
| WOUND-03 | Dead models excluded from actions, OC, movement | integration | `uv run pytest tests/test_wounds.py::test_eliminated_model_excluded -x` | ❌ Wave 0 |
| WOUND-05 | Episode terminates when all models on a side eliminated | integration | `uv run pytest tests/test_wounds.py::test_termination_on_elimination -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `just test`
- **Per wave merge:** `just validate`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/test_wounds.py` — new test file covering all 4 requirements
- [ ] May need a fixture for a wound-configured env (multi-wound models + opponents)

### Backward Compatibility Tests
| Scenario | What to Verify | Test Type |
|----------|---------------|-----------|
| YAML without `max_wounds` | Config parses, models get default=1, env runs | unit |
| 0-opponent config | `all_eliminated` doesn't vacuously trigger | integration |
| Existing test suite | All existing tests pass unchanged | regression (`just test`) |

## Open Questions

1. **Renderer treatment of dead models**
   - What we know: Dead models should be visually distinct. The renderer iterates all models.
   - What's unclear: Exact visual style (greyed out? X marker? semi-transparent?). This is aesthetic.
   - Recommendation: Use a greyed-out circle with reduced opacity or an X overlay. Claude's discretion per CONTEXT.md.

2. **Observation builder for dead models in Phase 1**
   - What we know: Phase 2 (WOUND-04) will add alive flags to observations. Phase 1 keeps shapes unchanged.
   - What's unclear: Should Phase 1 zero out observation fields for dead models, or leave real position data?
   - Recommendation: Leave real position data in Phase 1 (dead models just don't move, so position freezes). Phase 2 adds the alive flag so the policy can distinguish them.

## Sources

### Primary (HIGH confidence)
- Codebase direct inspection: `domain/entities.py`, `domain/termination.py`, `domain/battle.py`, `domain/battle_factory.py`, `domain/battle_view.py`
- Codebase direct inspection: `env_components/actions.py`, `env_components/distance_cache.py`, `env_components/observation_builder.py`
- Codebase direct inspection: `reward/phase_manager.py`, `reward/calculators/base.py`, `reward/calculators/closest_objective.py`, `reward/calculators/group_cohesion.py`
- Codebase direct inspection: `mission/vp_calculator.py`, `opponent/scripted_advance_to_objective_policy.py`, `renders/human.py`
- Codebase direct inspection: `wargame.py` (main env loop)
- Project design docs: `docs/ddd-envs.md`, `docs/tabletop-rules-reference.md`

### Secondary (MEDIUM confidence)
- CONTEXT.md user decisions (locked constraints for this phase)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new dependencies, purely internal changes
- Architecture: HIGH — codebase fully inspected, all iteration sites inventoried
- Pitfalls: HIGH — each pitfall identified from actual code patterns
- Test strategy: HIGH — existing test infrastructure is clear (pytest, conftest fixtures)

**Research date:** 2026-04-02
**Valid until:** 2026-05-02 (stable domain, no external dependencies)
