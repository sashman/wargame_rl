---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: Terrain & Line-of-Sight Blocking
status: planning
stopped_at: Phase 1 context gathered (footprint-based 10e Ruins LOS; walls deferred)
last_updated: "2026-06-20T10:16:25.950Z"
last_activity: 2026-06-20 — v2.0 roadmap created (2 phases, 10/10 TERR-* mapped)
progress:
  total_phases: 2
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-19)

**Core value:** Agents learn recognisable tactical behaviour through reward shaping and environment design
**Current focus:** v2.0 Terrain & Line-of-Sight Blocking — Phase 1 (Terrain in the Simulation)

## Current Position

Phase: 1 of 2 (Terrain in the Simulation — LOS-Blocking)
Plan: 0 of 2 in current phase
Status: Ready to plan
Last activity: 2026-06-20 — v2.0 roadmap created (2 phases, 10/10 TERR-* mapped)

Progress: [░░░░░░░░░░] 0%

**Roadmap shape:** Two phases with a clean one-way dependency.

- Phase 1 `01-terrain-los-blocking`: config + `domain/terrain.py` footprint model + endpoint-aware
  LOS seam + renderer (TERR-01..07, TERR-10). **Footprint-based 10e Ruins LOS**: footprint blocks
  when it sits between two outside models; see-into/see-out exceptions. `domain/los.py` Bresenham
  untouched. **Walls deferred** (revisit later). Context captured: `01-CONTEXT.md`.

- Phase 2 `02-terrain-observation`: terrain footprint entity-token stream through obs pipeline +
  both networks + PPO, `terrain_embedding` appended LAST and None-guarded (TERR-08, TERR-09).

**Phase dirs:** v1.0 phases archived to `.planning/milestones/v1.0-phases/`; v9.0 phase dir to
`.planning/milestones/v9.0-phases/`. v2.0 uses unprefixed `01-`/`02-` (tooling resolves correctly).

## Performance Metrics

**Velocity:**

- Total plans completed: 0
- Average duration: -
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**

- Last 5 plans: -
- Trend: -

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- **v2.0 build order:** Two phases, strictly sequential — Phase 2 needs `BattleView.terrain` from
  Phase 1. Phase 1 = mechanics (LOS), Phase 2 = perception (observation).

- **v2.0 wall representation:** WHOLE-CELL rasterisation OR'd into the existing `is_blocking(x,y)`
  seam; `domain/los.py` Bresenham unchanged; cell-edge "true thin" walls deferred (TERRF-01).

- **v2.0 footprint:** No-op area marker this milestone — NOT rasterised, no LOS effect.
- **v2.0 observation:** Terrain as a new entity-token stream appended LAST; `terrain_embedding` is
  `None` when `terrain_size == 0` (mirrors `opponent_model_embedding is None`) → no terrain =
  byte-identical behaviour, pre-terrain checkpoints still load.

- **Phase 3 (v1):** LOS uses Bresenham + interior-only `is_blocking`; optional `blocking_mask` in
  YAML; `domain/los.py` is the single source (terrain walls now merge into the same grid).

### Pending Todos

None yet.

### Blockers/Concerns

- **LOS symmetry (Phase 1):** Bresenham `A→B` vs `B→A` can diverge near cell corners — the Hypothesis
  symmetry property test is the highest-value test; canonicalise endpoint order in the seam if it fails.

- **L-wall rasterisation (Phase 1):** guarantee 4-connectivity to avoid diagonal pinholes / sealed
  gaps; verify with golden L-wall boards (elbow + intended gap).

- **Checkpoint compat (Phase 2):** verify pre-terrain checkpoint loads & infers; test Transformer +
  MLP + PPO; append terrain LAST, keep action_mask last in the tensor list.

- **Wall-camping / distribution shift (out of strict scope):** flag for the combat-reward / curriculum
  phase — keep a sparse objective anchor; ship terrain as a distinct config, not an in-place flag flip.

- CUDA setup may be broken on dev machine — use `CUDA_VISIBLE_DEVICES=""` only if training fails.

## Session Continuity

Last session: 2026-06-20T10:16:25.945Z
Stopped at: Phase 1 context gathered (footprint-based 10e Ruins LOS; walls deferred)
Resume file: .planning/phases/01-terrain-los-blocking/01-CONTEXT.md
