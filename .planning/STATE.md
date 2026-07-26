---
gsd_state_version: 1.0
milestone: v2.0
status: active
stopped_at: v2.0 complete — both phases shipped
last_updated: "2026-07-26"
last_activity: 2026-07-26
---

# Project State

**Core value:** Agents learn recognisable tactical behaviour through reward shaping and environment design
**Current focus:** v2.0 milestone complete

## Current Position

- **v2.0** (Terrain & LOS-Blocking): Both phases complete.
  - Phase 1 (LOS-Blocking): Shipped 2026-07-25
  - Phase 2 (Observation): Shipped 2026-07-26
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
- Terrain tokens appended LAST in transformer sequence (after opponents)
- `terrain_embedding` is None when `terrain_size == 0` (backward compat)
- Observation tensor pipeline: 6 tensors (game, obj, player, opp, terrain, mask)

## Blockers/Concerns

- CUDA setup may be broken on dev machine — use `CUDA_VISIBLE_DEVICES=""` for training

## Session Continuity

Last session: 2026-07-26
Stopped at: v2.0 milestone complete — both phases shipped
