# Phase 1: Wounds & Elimination - Context

**Gathered:** 2026-04-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Models gain durable wound state that changes during an episode. Models reduced to 0 wounds are eliminated — excluded from action selection, movement, and objective control. The episode terminates when all models on one side are eliminated. No damage source is introduced in this phase; shooting (Phase 5) will be the first real consumer of the wound system.

</domain>

<decisions>
## Implementation Decisions

### Wound Reduction Mechanism
- **D-01:** Add a `take_damage(amount)` method on `WargameModel` that reduces `current_wounds` (clamped to 0). This is the sole entry point for wound reduction across the codebase.
- **D-02:** No in-env damage source in Phase 1. Wounds are exercised via unit tests that call `take_damage` directly on domain objects during step sequences. Shooting resolution (Phase 5) will be the first real caller.

### Elimination Representation
- **D-03:** Flag-based elimination. Models stay in `player_models` / `opponent_models` lists at their original index. An `is_alive` property on `WargameModel` returns `current_wounds > 0`.
- **D-04:** Eliminated models are excluded from action selection, movement application, and objective control checks by filtering on `is_alive`. Array shapes never change mid-episode.
- **D-05:** All iteration over models that applies actions, checks OC, or computes distances must guard on `is_alive`. This includes `ActionHandler`, distance cache, and any reward calculators that inspect model positions.

### Default Wound Values
- **D-06:** Change `ModelConfig.max_wounds` default from 100 to 1 (standard tabletop infantry). Safe because no damage source exists until Phase 5; configs designed for combat will set explicit values.
- **D-07:** Existing YAML configs without `max_wounds` will get 1-wound models. This is acceptable — combat-era configs will be purpose-built with explicit wound values.

### Termination on Elimination
- **D-08:** Extend `is_battle_over` in `domain/termination.py` with an `all_eliminated` condition: episode ends when all player models or all opponent models are eliminated.
- **D-09:** This condition composes with existing termination (max turns, clock, all-at-objectives) — any condition being true ends the episode.

### Claude's Discretion
- Whether `take_damage` clamps at 0 or raises on negative input
- Internal implementation of alive-filtering helpers (property on Battle vs utility function)
- Test structure and parameterization for wound/elimination scenarios

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Domain structure and extension patterns
- `docs/ddd-envs.md` — DDD structure, dependency direction, how to add termination conditions and entity state
- `docs/tabletop-rules-reference.md` — Wound stat definition (W), damage/destruction rules in Attack Sequence section

### Existing wound scaffolding
- `wargame_rl/wargame/envs/domain/entities.py` — `WargameModel` with `stats` dict (`max_wounds`, `current_wounds`), `reset_for_episode()`
- `wargame_rl/wargame/envs/types/config.py` — `ModelConfig.max_wounds` field, `WargameEnvConfig` validation
- `wargame_rl/wargame/envs/domain/battle_factory.py` — How models are constructed with stats from config
- `wargame_rl/wargame/envs/domain/battle.py` — `Battle` aggregate, `reset_for_episode()`

### Termination
- `wargame_rl/wargame/envs/domain/termination.py` — `is_battle_over` function to extend
- `wargame_rl/wargame/envs/wargame.py` — Where `is_battle_over` is called in `step()`

### Action masking and model iteration
- `wargame_rl/wargame/envs/env_components/actions.py` — `ActionHandler` that applies actions per model
- `wargame_rl/wargame/envs/env_components/distance_cache.py` — Distance computation that needs alive filtering
- `wargame_rl/wargame/envs/domain/battle_view.py` — `BattleView` protocol (may need alive-related properties)

### Requirements
- `.planning/REQUIREMENTS.md` — WOUND-01 through WOUND-05 requirements for this phase

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `WargameModel.stats` dict already stores `max_wounds` and `current_wounds` — no new data structure needed
- `WargameModel.reset_for_episode()` already resets `current_wounds = max_wounds`
- `ModelConfig.max_wounds` already exists as a configurable field
- `WargameModel.to_space()` already defines observation space for wounds stats
- `Battle.reset_for_episode()` already iterates all models calling their reset

### Established Patterns
- Domain entities own their state and mutation methods (e.g. `set_previous_closest_objective_distance`)
- `is_battle_over` uses a flat function signature with boolean flags — extend with an `all_eliminated` parameter
- `BattleView` protocol exposes read-only properties consumed by reward/render — add alive-related views here
- Registry pattern for calculators/criteria — any new reward calculator for elimination would follow this

### Integration Points
- `WargameEnv.step()` calls `is_battle_over` — needs to compute and pass the elimination flag
- `ActionHandler` iterates models to apply actions — needs alive guard
- `compute_distances` in step — needs to filter or handle eliminated models
- `build_observation` — Phase 2 will add alive flags to obs; Phase 1 data structure must support it

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches following existing DDD patterns.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-wounds-elimination*
*Context gathered: 2026-04-02*
