---
gsd_state_version: 1.0
milestone: between
status: active
stopped_at: v9.0 shipped; planning next milestone
last_updated: "2026-07-25"
last_activity: 2026-07-25
---

# Project State

**Core value:** Agents learn recognisable tactical behaviour through reward shaping and environment design
**Current focus:** Between milestones — next up: v2.0 (Terrain) or v7.0 (Self-Play)

## Current Position

- **v9.0** (Structured State): ✅ Shipped 2026-06-19. Archive in `milestones/`.
- **v1.0** (Ranged Combat): Phases 1–5 complete. Phase 6 deferred.
- **Next:** `/gsd-new-milestone` to start v2.0 or v7.0.

## Decisions

Key decisions logged in PROJECT.md. Architecture decisions that persist:
- Snapshot schema uses native Python types (no numpy) for trivial serialisation
- Event log uses anchor snapshots at configurable intervals for efficient seek
- LOS: Bresenham + interior-only blocking; single `domain/los.py` source
- Shooting resolution at env level (not ActionHandler) — env owns combat flow

## Blockers/Concerns

- CUDA setup may be broken on dev machine — use `CUDA_VISIBLE_DEVICES=""` for training

## Session Continuity

Last session: 2026-07-25
Stopped at: Compacted .planning/ artifacts
