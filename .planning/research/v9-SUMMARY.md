# Research Summary: v9.0 — Structured Game State & LLM-Readable Representation

**Project:** Wargame RL
**Milestone:** v9.0
**Researched:** 2026-05-23
**Confidence:** HIGH (all findings from direct codebase analysis)

## Executive Summary

The wargame_rl codebase already contains rich structured data at every step — entity positions and wounds, game clock state, combat results via `ShootingResult`, detailed reward breakdowns, and action indices that can be decoded to human-readable text. However, none of this data is surfaced outside the RL tensor pipeline. The Gymnasium info dict (`WargameEnvInfo`) captures roughly half the state, while critical data — game timing, combat results, actions taken, reward breakdowns, objective control — is either computed and discarded or stored in private env fields without serialisation. The gap is not missing data but missing *output plumbing*.

The recommended approach is to build a Pydantic-based canonical state model (`GameStateSnapshot`) that collects all game state into a single serialisable object, then implement bidirectional I/O: *export* (env → snapshot → JSON) and *injection* (JSON → snapshot → env). Pydantic is the right tool — it's already the config standard, provides `model_dump_json()` and `model_json_schema()` for free, handles numpy serialisation via `mode="json"`, and composes nested models naturally. The new layer fits cleanly as a `BattleView` consumer alongside renderers and reward calculators, with zero impact on the RL pipeline.

The main risk is scope creep. The user's priorities are **(1) bidirectional state I/O** and **(2) LLM-interpretable representation**. Event streaming, replay, delta encoding, and pluggable serialisation codecs are documented in the v9.0 requirements but are lower priority and should be deferred to later phases. The other key risk is the `GameClock` — it lacks a `set_state()` method and all its fields are private, making state injection impossible without a small domain change.

## Key Findings

### State Representation (v9-STATE-REPRESENTATION.md)

The full game state spans five sources: `Battle` aggregate (models, objectives, VP, zones), `GameClock` (round, phase, active player), `WargameEnv` ephemeral fields (turn counter, reward, shooting results, combat RNG), `WargameModel` entities (location, wounds, distances, reward-shaping history), and `WargameEnvConfig` (immutable rules). A comprehensive gap analysis found **17 data categories** missing from the info dict that are present internally, including battle round/phase, terrain/LOS, actions taken, shooting results, reward breakdown, objective ownership, and episode context.

**Key conclusion:** Pydantic v2 is highly leverageable. `model_dump(mode="json")` auto-converts numpy arrays to lists. `model_json_schema()` can generate JSON Schema for LLM tool-use integration. JSON is the recommended primary format — LLMs work natively with it, and schema generation is free.

### LLM Representation (v9-LLM-REPRESENTATION.md)

An LLM evaluator needs a structured per-step summary covering: state before action, decoded action descriptions, combat narrative with probabilities, reward breakdown with phase context, and state after. The codebase has the data but lacks text formatting.

**What exists but isn't surfaced:**
- `ActionHandler._decode_action()` returns (dx, dy) — needs a `describe_action()` wrapper for "Move NE at speed 4"
- `ShootingResult` stores hits/wounds/unsaved/damage — but not attacker-target pairing
- `last_reward_breakdown` dict with per-calculator values — never serialised
- Opponent actions are applied and discarded — never recorded
- Objective control is computed for rendering only — not in observations

**What needs building:** `describe_action()` text formatter, attacker-target pairing in shooting results, opponent action recording, objective control in state output, `StepNarrator` class that produces LLM-readable text from `BattleView` + step context.

### Bidirectional State I/O (v9-STATE-IO.md)

**Export (state out) is straightforward.** All mutable state is readable via properties or public attributes. `GameClock.state` already returns a `GameState` snapshot. No private internals need exposure.

**Injection (state in) is the hard part.** The env has no "set state" path. Three things block it:
1. `GameClock` has no `set_state()` method — all fields are private with no setters
2. No `Battle.apply_snapshot()` — entity state must be set via direct mutation (tests already do this)
3. No `WargameEnv.load_state()` orchestrator — nothing coordinates clock + entities + VP + derived state recomputation

Derived state (distances_to_objectives, action masks, distance cache) should be recomputed after injection, not injected. Reward-shaping history (`previous_closest_objective_distance`, `best_closest_objective_distance`) should be set to `None` — the safest sentinel for "first step after injection." Existing test patterns (post-reset mutation of locations and wounds) validate that direct entity mutation works.

### Architecture (v9-ARCHITECTURE.md)

The state layer fits cleanly into the DDD architecture as `envs/state/`, a peer of `envs/renders/` and `envs/reward/`. It depends on `BattleView` + `types/` only — same dependency direction as every other consumer. The registry pattern used by reward calculators, criteria, missions, and opponent policies can be reused for formatter registration. Hooks attach after `reset()` and `step()`, mirroring the renderer pattern.

**Zero impact on RL pipeline.** `observation_to_tensor()`, Lightning modules, `train.py`, and checkpoints are untouched. State export is a parallel output path serving different consumers (LLMs, APIs, debugging) than the tensor pipeline (RL networks).

**Zero impact on existing configs.** New `WargameEnvConfig` fields default to `None` or empty list, so existing YAML files keep working.

## Implications for Roadmap

Based on research, I recommend **four phases** with the first two being the user's stated priorities and the latter two being deferrable.

### Phase 1: Canonical State Model & Export

**Rationale:** Everything depends on the canonical state schema. Export (state → JSON) is the simpler direction and immediately enables programmatic state inspection, LLM consumption via JSON, and debugging. This is the foundation for both user priorities.

**Delivers:**
- `GameStateSnapshot` Pydantic model with all fields from the state inventory
- `ModelSnapshot`, `ObjectiveSnapshot`, `ClockSnapshot` sub-models
- numpy → JSON serialisation via `@field_serializer` or `mode="json"`
- `WargameEnv.to_snapshot() → GameStateSnapshot` method
- `GameStateSnapshot.model_dump_json()` for JSON output
- `GameStateSnapshot.model_json_schema()` for schema export
- Combat result capture with attacker-target pairing (modify `_resolve_shooting_action`)
- Opponent action recording (store before applying)
- Objective control computation included in snapshot
- Reward breakdown and phase name in snapshot
- `schema_version` field from day one

**From research:** STATE-REPRESENTATION §6 (proposed structure), ARCHITECTURE §3 (lives in `envs/state/`), ARCHITECTURE §6 (parallel path to RL pipeline)

**Avoids:** Performance overhead — export is optional, disabled by default during training. Only active during simulation/evaluation/API serving.

### Phase 2: State Injection & Bidirectional I/O

**Rationale:** Depends on Phase 1's schema. This is the "input" side of bidirectional I/O — the ability to construct an env at an arbitrary game state from a snapshot. Required for scenario testing, LLM-driven evaluation, and programmatic game setup.

**Delivers:**
- `GameClock.set_state()` method (small domain change, 5-10 lines)
- `validate_snapshot(snapshot, config)` function enforcing hard constraints
- `WargameEnv.load_state(snapshot)` orchestrator that sets clock, entities, VP, recomputes derived state
- Round-trip test: `env.to_snapshot()` → `env.load_state(snapshot)` → `env.to_snapshot()` produces identical output
- Documented invariants (locations in bounds, wounds in range, clock consistency, entity count match)

**From research:** STATE-IO §3–4 (injection requirements, clock gaps), STATE-IO §6 (validation constraints), STATE-IO §7 (test patterns)

**Avoids:** Reward-shaping state discontinuity — sets shaping fields to `None` (first-step sentinel), which is the safest behaviour.

### Phase 3: LLM-Readable Text Representation

**Rationale:** Depends on Phase 1's snapshot model and action/combat data capture. Builds the human/LLM-readable text layer on top of the canonical state. This is the "LLM-interpretable" priority — an LLM can read structured text and judge whether the RL agent's actions are good or bad.

**Delivers:**
- `describe_action(action_int, model_idx, phase) → str` method on `ActionHandler`
- Compass direction mapping (16-angle → direction name)
- `CombatNarrative` structured type with probabilities (hit chance, wound threshold, modified save)
- `StepNarrator` class producing per-step text summaries from `BattleView` + step context
- Reward breakdown text with phase context (essential — without phase name, reward values are uninterpretable)
- `text` formatter registered in formatter registry

**From research:** LLM-REPRESENTATION §1 (action decoding), §3 (step summary requirements), §6 (combat narrative data), §8 (implementation priorities)

### Phase 4: Event Streaming, Replay & Advanced Serialisation (Deferred)

**Rationale:** Lower priority per user's stated preferences. Depends on stable snapshot model from Phase 1. Can be built incrementally after the core I/O and LLM features are proven.

**Delivers (when needed):**
- `StateExporter` protocol with `on_reset()` / `on_step()` hooks
- Append-only event log (ordered events recording match history)
- Delta encoding (efficient state updates vs full snapshots)
- Deterministic replay from event log
- Pluggable formatter registry (JSON is default; extension points for binary/compact formats)
- `EventLog` accumulator as an exporter implementation

**From research:** ARCHITECTURE §4 (registry pattern), §7 (hook attachment points), §8 (BattleView data availability)

### Phase Ordering Rationale

- **Phase 1 before Phase 2:** Export is simpler than injection (read vs write), and the schema must exist before injection can target it. Export also provides immediate value for debugging and LLM experiments.
- **Phase 2 before Phase 3:** Bidirectional I/O is the user's top priority. LLM text is built on top of the same data but adds a formatting layer that's less architecturally critical.
- **Phase 3 after Phase 2:** The `describe_action()` and `CombatNarrative` types from Phase 3 enhance the snapshot from Phase 1, but the snapshot is useful in JSON form without text narration. Phase 3 can proceed independently once Phase 1 is complete.
- **Phase 4 is deferred:** Event streaming and replay require a stable snapshot schema. Building them too early risks rework when the schema evolves during Phases 1–3.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 2:** State injection is genuinely new capability — no existing tests construct mid-episode state. The `GameClock.set_state()` design needs careful validation against clock invariants (round boundaries, phase ordering, active player transitions).
- **Phase 3:** LLM text format needs validation against actual LLM tool-use patterns. The proposed text format (structured plaintext vs JSON) should be tested with a real LLM to confirm interpretability.

Phases with standard patterns (skip research-phase):
- **Phase 1:** Well-documented Pydantic patterns, clear data sources, straightforward read-only serialisation. The research already provides the full field inventory and proposed model shape.
- **Phase 4:** Registry pattern is established (4 existing registries), hook pattern mirrors renderer. Standard implementation.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| State representation | HIGH | Complete field inventory from code inspection; every field traced to source |
| LLM representation | HIGH for data availability, MEDIUM for text format | Data exists; optimal LLM consumption format needs validation with actual LLM usage |
| State I/O (export) | HIGH | All state readable via properties; `GameClock.state` already returns snapshot |
| State I/O (injection) | HIGH for feasibility, MEDIUM for edge cases | Tests show direct mutation works; clock injection needs new method; reward-shaping discontinuity is documented |
| Architecture | HIGH | DDD layers, BattleView protocol, registry pattern all well-established; new layer fits naturally |

**Overall confidence:** HIGH

### Gaps to Address

- **LLM text format validation:** The proposed structured text format should be tested with an actual LLM before committing to it as the canonical format. JSON may be sufficient for LLM tool-use, making a custom text format a nice-to-have rather than essential.
- **Coordinate system in narration:** The grid uses screen coordinates (+y = south) but angles start at east going counter-clockwise. The narration layer must document this and map correctly to compass directions. Bugs here would produce misleading LLM evaluations.
- **numpy serialisation strategy:** `mode="json"` auto-converts numpy arrays to lists, but large arrays (e.g., 50×50 blocking mask) may need compact encoding. Need to decide: always convert to lists, or use a compact representation for terrain.
- **Schema versioning strategy:** `schema_version` should be included from day one (Phase 1). The versioning policy (breaking vs non-breaking changes, deprecation) should be documented but doesn't need to be fully designed upfront.
- **Performance budget:** State export during training should be opt-in and benchmarked. The exporter should add negligible overhead when disabled. During evaluation/simulation, JSON serialisation of a typical state (4 models, 4 opponents, 3 objectives) should be sub-millisecond.

## Sources

### Primary (HIGH confidence)
- Direct codebase analysis — all findings verified against source code
- `domain/battle.py`, `domain/entities.py`, `domain/game_clock.py`, `domain/value_objects.py` — state inventory
- `envs/wargame.py` — env lifecycle, reset flow, step flow, BattleView implementation
- `envs/types/` — config, observation, info, action types
- `envs/env_components/` — action handler, observation builder, placement
- `envs/reward/` — phase manager, calculators, criteria
- `docs/ddd-envs.md` — architecture guide and extension patterns

### Secondary (MEDIUM confidence)
- Pydantic v2 documentation — `model_dump_json()`, `model_json_schema()`, `@field_serializer` capabilities
- LLM tool-use patterns — JSON Schema as function parameter schema (well-established pattern, not project-specific)

---
*Research completed: 2026-05-23*
*Ready for roadmap: yes*
