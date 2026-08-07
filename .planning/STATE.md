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
**Current focus:** v2.0 complete. Phase 03 (continuous space) planned and ready to start.

## Current Position

- **Phase 03** (Continuous Space & Model Geometry): **planned, not started**. Precedes the planned v3.0 (advanced movement), which depends on it. A working
  prototype exists on `feature/continuous-positions-and-bases` (`168f036`, `db4a4dd`,
  `3c7812e`) and was built to be learned from rather than merged. The plan it produced
  is `phases/03-continuous-space-and-geometry/03-PLAN.md`, and the measured findings
  are in `reports/2026-08-07-continuous-space-and-model-bases.md`.
- **v2.0** (Terrain & LOS-Blocking): Both phases complete.
  - Phase 1 (LOS-Blocking): Shipped 2026-07-25
  - Phase 2 (Observation): Shipped 2026-07-26
- **v9.0** (Structured State): Shipped 2026-06-19.
- **v1.0** (Ranged Combat): Phases 1–5 complete. Phase 6 deferred.

## Decisions

Key v2 decisions (from `phases/01-terrain-los-blocking/01-CONTEXT.md`):
- **Footprint-based Ruins LOS** — no walls this milestone
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
- **Every baseline recorded before Phase 03 is void.** The prototype moved the
  `squad_march_shoot` bar on `25v25_cover_control` from 0.45 to 1.00 at the same weapon
  range. Re-measure before comparing anything across that boundary.

## Session Continuity

Last session: 2026-08-07
Stopped at: Phase 03 planned from a prototype. WP-0 (deleting the two quantisations) is
independent of the rest and can ship on its own — it is the only change in the milestone
that improves the environment without shifting the dynamics, so it is the only one whose
effect on learning can be measured cleanly.
