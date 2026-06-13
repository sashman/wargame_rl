# Roadmap: Structured Game State & LLM-Readable Representation

## Overview

This milestone adds programmatic state I/O and LLM-interpretable representation to the wargame environment. The codebase already contains rich structured data at every step — entity positions and wounds, game clock state, combat results, detailed reward breakdowns, and decodable action indices — but none of it is surfaced outside the RL tensor pipeline. This milestone builds a Pydantic-based canonical state model (`GameStateSnapshot`), bidirectional I/O (export and injection), and a text narration layer for LLM evaluation.

The build order follows dependency: the canonical schema first (everything depends on it), then state injection (the "input" side of bidirectional I/O), then LLM text (builds on the data captured in the snapshot). Event streaming and replay build on the stable schema established in Phases 1–3. A final validation phase verifies all milestone requirements and success criteria end-to-end.

## Phases

- [x] **Phase 1: Canonical State Model & Export** — Pydantic `GameStateSnapshot` projected from `BattleView`, JSON serialisation, `WargameEnv.to_snapshot()`, combat/action metadata capture (completed 2026-05-23)
- [x] **Phase 2: State Injection & Bidirectional I/O** — `GameClock.set_state()`, `WargameEnv.load_state(snapshot)`, validation, derive computed fields from injected state (completed 2026-05-23)
- [x] **Phase 3: LLM-Readable Text Representation** — Action descriptions, combat narrative, step narrator, reward breakdown text for LLM evaluation (completed 2026-05-23)
- [ ] **Phase 4: Event Streaming & Replay** — Append-only event log, delta encoding, deterministic replay, pluggable codec interface
- [ ] **Phase 5: Milestone Validation** — End-to-end verification of all v9 requirements and success criteria across Phases 1–4

## Phase Details

### Phase 1: Canonical State Model & Export
**Goal**: The full game state is available as a structured, serialisable Pydantic model that can be exported to JSON at any point during an episode
**Depends on**: Nothing (first phase)
**Requirements**: SGS-01, SGS-04, SGS-11, SGS-12, SGS-14
**Success Criteria** (what must be TRUE):
  1. `WargameEnv.to_snapshot()` returns a `GameStateSnapshot` Pydantic model containing board, entities, clock, VP, combat results, actions taken, and reward breakdown
  2. `snapshot.model_dump_json()` produces valid JSON with all numpy arrays converted to native Python types — parseable by any JSON consumer
  3. `GameStateSnapshot.model_json_schema()` produces a JSON Schema document describing the state structure (usable as LLM tool parameter schema)
  4. Combat results include attacker-target pairing (which model shot which target) — not just flat `ShootingResult` lists
  5. Opponent actions are recorded before application and available in the snapshot
**Plans:** 1 plan
Plans:
- [x] v9-01-01-PLAN.md — Snapshot Pydantic models, PairedShootingResult, build_snapshot factory, to_snapshot(), encoder protocol, tests

### Phase 2: State Injection & Bidirectional I/O
**Goal**: An arbitrary game state can be constructed from a snapshot, enabling scenario testing, LLM-driven evaluation, and programmatic game setup
**Depends on**: Phase 1
**Requirements**: SGS-07, SGS-08, SGS-09
**Success Criteria** (what must be TRUE):
  1. `WargameEnv.load_state(snapshot)` sets clock, entity positions/wounds, VP, and recomputes derived state (distances, action masks) — the env is ready for `step()` immediately after
  2. `GameClock.set_state()` can position the clock at any valid round/phase/player combination
  3. Round-trip fidelity: `env.to_snapshot()` → `env.load_state(snapshot)` → `env.to_snapshot()` produces identical output (excluding derived-only fields)
  4. `validate_snapshot(snapshot, config)` rejects snapshots that violate hard constraints (locations out of bounds, wounds out of range, entity count mismatch, invalid clock state)
**Plans**: Implemented directly (no formal plan — code landed in v9-01 branch)
**Key commits**: `42fad10` — `GameClock.set_state()`, `validate_snapshot()`, `WargameEnv.load_state()`, 23 tests

### Phase 3: LLM-Readable Text Representation
**Goal**: An LLM can read a structured text summary of any game step and judge whether the RL agent's actions are tactically sound
**Depends on**: Phase 1
**Requirements**: SGS-02, SGS-10, SGS-13
**Success Criteria** (what must be TRUE):
  1. `describe_action(action_int, model_idx, phase)` returns human-readable text (e.g. "Move NE at speed 4" or "Shoot at opponent 1") with correct compass direction mapping for all 16 angles
  2. `StepNarrator` produces per-step text summaries covering: state before action, decoded actions, combat narrative with hit/wound/save probabilities, reward breakdown with phase context, and state after
  3. Combat narrative includes analytical context: expected damage per target, wound threshold, modified save — enough for an LLM to evaluate whether the agent chose the optimal target
  4. Reward breakdown text includes the active reward phase name — without this, reward values are uninterpretable to an LLM
**Plans**: Implemented directly (no formal plan — code landed in v9-01 branch)
**Key commits**: `ee30c1b` — `StepNarrator`, public `describe_action` API, 8 tests

### Phase 4: Event Streaming & Replay
**Goal**: A complete match history can be recorded as an ordered event stream and replayed deterministically
**Depends on**: Phase 1 (stable schema — now complete)
**Requirements**: SGS-03, SGS-05, SGS-06
**Success Criteria** (what must be TRUE):
  1. A `StateExporter` protocol with `on_reset()` / `on_step()` hooks produces an append-only event log recording the full match
  2. Delta encoding represents state updates efficiently (full snapshots at anchors, granular deltas between them)
  3. Deterministic replay: applying the event log from a known initial configuration reconstructs any requested historical state
  4. A pluggable formatter registry (extending SGS-04) allows alternative encodings behind a shared codec interface
**Plans**: TBD

### Phase 5: Milestone Validation
**Goal**: All v9 requirements and phase success criteria are verified end-to-end against the live codebase
**Depends on**: Phases 1–4
**Requirements**: All SGS-* requirements
**Success Criteria** (what must be TRUE):
  1. Every SGS-* requirement marked Complete has a passing test or verifiable evidence
  2. All phase success criteria (Phases 1–4) hold against the current codebase
  3. Round-trip and replay scenarios exercise the full pipeline: snapshot → inject → step → stream → replay
  4. Requirements traceability table in REQUIREMENTS.md is fully updated
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5
(Phases 2 and 3 are independent after Phase 1 and can execute in parallel)

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Canonical State Model & Export | 1/1 | Complete | 2026-05-23 |
| 2. State Injection & Bidirectional I/O | 1/1 | Complete | 2026-05-23 |
| 3. LLM-Readable Text Representation | 1/1 | Complete | 2026-05-23 |
| 4. Event Streaming & Replay | 0/0 | Not started | - |
| 5. Milestone Validation | 0/0 | Not started | - |
