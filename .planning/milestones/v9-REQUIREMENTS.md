# Requirements Archive — v9.0: Structured Game State & Event Streaming

**Status:** ✅ SHIPPED 2026-06-19
**Roadmap:** `.planning/milestones/v9-ROADMAP.md`
**Core Value:** Agents learn recognisable tactical behaviour through reward shaping and environment design

This archive captures the v9.0 milestone requirements (structured state & events), all
complete with test evidence. The v9 requirements were extracted from the shared
`.planning/REQUIREMENTS.md` at milestone completion; the v1.0 (Shooting & Destruction)
requirements remain in the active `REQUIREMENTS.md` because that milestone has deferred work
(Phase 6: Combat Reward & Curriculum).

## Requirements

### Canonical State & Export

- [x] **SGS-01**: A canonical programmatic game-state model exists (board, entities, phase, scoring, etc.), sourced from domain / read-only views, not tied to RL observation tensors
- [x] **SGS-02**: Serialised state is suitable for external APIs and for LLM-facing validation (stable identifiers, documented semantics, explicit schema version)
- [x] **SGS-04**: Default encoding is JSON; a codec or encoder interface allows additional formats without changing the canonical model

### State Injection (discovered during v9.0 research)

- [x] **SGS-07**: `GameClock.set_state()` can position the clock at any valid round/phase/player combination for state injection
- [x] **SGS-08**: `WargameEnv.load_state(snapshot)` constructs a mid-episode environment from a `GameStateSnapshot`, recomputing derived state
- [x] **SGS-09**: Round-trip state fidelity: `to_snapshot()` → `load_state()` → `to_snapshot()` produces identical output

### LLM Text Representation (discovered during v9.0 research)

- [x] **SGS-10**: `describe_action()` produces human/LLM-readable text for any action integer (e.g. "Move NE at speed 4", "Shoot at opponent 1")
- [x] **SGS-11**: Combat results include attacker-target pairing and analytical context (probabilities, expected damage) for LLM evaluation
- [x] **SGS-12**: Opponent actions are recorded before application and available in state output
- [x] **SGS-13**: `StepNarrator` produces per-step text summaries covering state, decoded actions, combat narrative, and reward breakdown with phase context
- [x] **SGS-14**: Reward breakdown and active reward phase name are included in state output

### Event Streaming & Replay

- [x] **SGS-03**: A layered change protocol expresses updates as full snapshots and/or granular deltas at defined abstraction levels to minimise redundancy
- [x] **SGS-05**: An append-only, ordered event stream can represent a complete match history for storage or streaming
- [x] **SGS-06**: Replay is deterministic: events (optionally with periodic snapshots) applied from a known initial configuration reconstruct any requested historical state (fast-forward / seek)

## Traceability

| Requirement | Phase | Status | Evidence |
|-------------|-------|--------|----------|
| SGS-01 | Phase 1 | Complete | `test_snapshot.py`, `test_v9_milestone_validation.py::test_sgs01` |
| SGS-02 | Phase 3 | Complete | `test_snapshot.py::TestJsonSerialisation`, `test_v9_milestone_validation.py::test_sgs02` |
| SGS-03 | Phase 4 | Complete | `test_event_stream.py::TestDeltaEncoding`, `test_v9_milestone_validation.py::test_sgs03` |
| SGS-04 | Phase 1 | Complete | `test_snapshot.py::TestEncoder`, `test_event_stream.py::TestCodecRoundTrip`, `test_v9_milestone_validation.py::test_sgs04` |
| SGS-05 | Phase 4 | Complete | `test_event_stream.py::TestEventLog`, `test_v9_milestone_validation.py::test_sgs05` |
| SGS-06 | Phase 4 | Complete | `test_event_stream.py::TestReplay`, `test_v9_milestone_validation.py::test_sgs06` |
| SGS-07 | Phase 2 | Complete | `test_state_injection.py::TestClockSetState`, `test_v9_milestone_validation.py::test_sgs07` |
| SGS-08 | Phase 2 | Complete | `test_state_injection.py::TestLoadState`, `test_v9_milestone_validation.py::test_sgs08` |
| SGS-09 | Phase 2 | Complete | `test_state_injection.py::TestRoundTrip`, `test_v9_milestone_validation.py::test_sgs09` |
| SGS-10 | Phase 3 | Complete | `test_narrator.py::TestDescribeActionPublic`, `test_v9_milestone_validation.py::test_sgs10` |
| SGS-11 | Phase 1 | Complete | `test_snapshot.py::TestCombatResults` |
| SGS-12 | Phase 1 | Complete | `test_snapshot.py` (opponent actions) |
| SGS-13 | Phase 3 | Complete | `test_narrator.py` (narrate tests), `test_v9_milestone_validation.py::test_sgs13` |
| SGS-14 | Phase 1 | Complete | `test_snapshot.py::TestRewardBreakdown`, `test_v9_milestone_validation.py::test_sgs14` |

**Coverage:**
- v9.0 requirements: 14 total (6 original + 8 discovered during research)
- Complete: 14 ✓
- Unmapped: 0
- End-to-end pipeline validated: `test_v9_milestone_validation.py::TestEndToEndPipeline`

---
*Archived: 2026-06-19 — v9.0 milestone complete, all 14 SGS-* requirements verified with test evidence.*
