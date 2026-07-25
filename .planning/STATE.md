---
gsd_state_version: 1.0
milestone: v2.0
status: active
stopped_at: v2.0 planning complete; Phase 1 ready to execute
last_updated: "2026-07-25"
last_activity: 2026-07-25
---

# Project State

**Core value:** Agents learn recognisable tactical behaviour through reward shaping and environment design
**Current focus:** v2.0 Phase 1 — Terrain in the Simulation (LOS-Blocking)

## Current Position

- **v2.0** (Terrain & LOS-Blocking): Phase 1 planned (2 plans ready), not started.
- **v9.0** (Structured State): Shipped 2026-06-19.
- **v1.0** (Ranged Combat): Phases 1–5 complete. Phase 6 deferred.

## Decisions

Key v2 decisions (from `phases/01-terrain-los-blocking/01-CONTEXT.md`):
- **Footprint-based 10e Ruins LOS** — no walls this milestone
- Blocking is **endpoint-aware** (per-query predicate, not a static grid)
- See-into/see-out exceptions evaluated **per ruin**
- `domain/los.py` Bresenham core stays **untouched**
- Canonicalise endpoint order in seam for symmetry guarantee
- Movement unaffected by terrain

## Blockers/Concerns

- CUDA setup may be broken on dev machine — use `CUDA_VISIBLE_DEVICES=""` for training
- Verify `hypothesis` is available as dev dep before executing Phase 1

## Session Continuity

Last session: 2026-07-25
Stopped at: Reconciled v2 planning onto compacted .planning/
