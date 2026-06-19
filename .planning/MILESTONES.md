# Milestones

Shipped milestones for Wargame RL. Each entry links to its archived roadmap and requirements
in `.planning/milestones/`.

---

## v9.0 — Structured Game State & LLM-Readable Representation

**Shipped:** 2026-06-19
**Phases:** 5 | **Plans:** 3 formal (Phases 2–4 implemented directly) | **Requirements:** 14/14 SGS-* complete
**PRs:** #110, #111, #114, #116
**Archive:** [v9-ROADMAP.md](milestones/v9-ROADMAP.md) · [v9-REQUIREMENTS.md](milestones/v9-REQUIREMENTS.md)

**Delivered:** A canonical, serialisable game-state model with bidirectional I/O, LLM-readable
text narration, and an append-only event stream with deterministic replay — surfacing the
environment's rich internal state for external APIs, scenario authoring, and LLM evaluation.

**Key accomplishments:**
1. Pydantic `GameStateSnapshot` projected from `BattleView`, exportable to JSON with a JSON Schema document (`WargameEnv.to_snapshot()`)
2. Bidirectional I/O — `GameClock.set_state()` + `WargameEnv.load_state()` with `validate_snapshot()` and verified round-trip fidelity
3. `StepNarrator` and public `describe_action()` produce LLM-readable per-step text with combat narrative, probabilities, expected damage, and reward-phase context
4. Append-only event stream with delta encoding and deterministic replay behind a pluggable codec interface (JSONL)
5. End-to-end milestone validation: 25 validation tests covering the full snapshot → inject → step → stream → replay pipeline

**Stats:** new `envs/state/` module (~2,040 LOC, 9 files); ~1,615 LOC of tests; ~27 days (2026-05-23 → 2026-06-19).
