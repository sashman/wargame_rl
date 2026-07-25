# Roadmap: Terrain & Line-of-Sight Blocking (v2.0)

## Milestones

- v9.0 Structured Game State — shipped 2026-06-19
- v1.0 Shooting & Model Destruction — complete (Phase 6 deferred)
- **v2.0 Terrain & Line-of-Sight Blocking** — active (this roadmap)

## Overview

This milestone adds terrain (Ruins) that blocks **line of sight only** to the existing PPO/DQN
wargame env, using the canonical Warhammer 40k 10e **Ruins abstraction**: a terrain piece is a
**footprint rectangle** and the *footprint itself* is the LOS blocker. A ruin blocks the line between
two models only when its footprint lies between them and **both** are outside it; a model inside the
footprint can **see out** and be **seen into** (per-ruin see-into/see-out exceptions). Walls have **no
LOS role** this milestone (deferred). Movement is untouched. Terrain is **encoded in the observation**.
This is a pure domain-logic + NumPy + existing-Pydantic-config change with **no new dependencies**,
extending surfaces the codebase already has (the single Bresenham LOS seam, the entity-token
transformer, the config→factory→aggregate→view DDD spine).

The work splits into a natural **two-phase build order** with a clean one-way dependency: Phase 1
puts terrain into the simulation/LOS (does the rule work?); Phase 2 puts terrain into the observation
(can the agent see it?). Phase 2 needs `BattleView.terrain` from Phase 1. The footprint-based LOS is
**endpoint-aware** — a cell is blocking iff it is inside some footprint F with both query endpoints
outside F — built as a per-query `is_blocking(x, y)` predicate that plugs into the existing seam; the
Bresenham core in `domain/los.py` stays untouched. The central backward-compat guarantee: **no terrain
⇒ zero terrain tokens ⇒ byte-identical behaviour and pre-terrain checkpoints still load.**

> **LOS model note:** An earlier draft assumed thin walls rasterised to blocking cells. Phase 1
> discussion reset to the source 10e Ruins rules, which abstract LOS to the **footprint** (you cannot
> see through a ruin even via windows). Walls are deferred. See `01-terrain-los-blocking/01-CONTEXT.md`.

> **Phase directory naming:** This milestone uses unprefixed directories
> (`01-terrain-los-blocking`, `02-terrain-observation`). Prior milestone phases were removed
> during planning compaction (git history preserves them).

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Terrain in the Simulation (LOS-Blocking)** - Config + domain footprint model + endpoint-aware LOS seam + renderer so ruin footprints block line of sight (with see-in/see-out) while movement is unaffected
- [ ] **Phase 2: Terrain in the Observation** - Terrain entity-token stream through the obs pipeline and both networks so the agent can see and reason about terrain

## Phase Details

### Phase 1: Terrain in the Simulation (LOS-Blocking)
**Goal**: Ruin footprints authored in YAML block line of sight through the single LOS service (with
10e see-into/see-out exceptions) while movement is unaffected; configs with no terrain behave exactly
as today.
**Directory**: `01-terrain-los-blocking`
**Depends on**: Nothing (first phase; depends only on the existing domain/LOS layer)
**Requirements**: TERR-01, TERR-02, TERR-03, TERR-04, TERR-05, TERR-06, TERR-07, TERR-10
**Success Criteria** (what must be TRUE):
  1. A YAML config can declare a terrain piece as a footprint rectangle `[x0,y0,x1,y1]` in absolute coords; loading is rejected with a clear error when a corner is off-board or two footprints overlap.
  2. With both models outside a footprint that lies between them, LOS is blocked; if the observer or the target is inside the footprint, LOS is clear (see-out / see-into); a footprint not on the line never blocks — verified by golden boards.
  3. LOS is symmetric (A sees B iff B sees A) and deterministic on known boards — verified by a Hypothesis property test over randomised footprints and endpoints.
  4. Shooting masks, action masks, shooting resolution, the renderer overlay, and the snapshot all agree on the same footprint blocking because they route through the single `has_line_of_sight_between_cells` seam (an endpoint-aware blocking predicate); the renderer colours the debug LOS line by the actual verdict.
  5. Models can move through and occupy footprint cells (movement unaffected), and an existing no-terrain config produces byte-identical LOS behaviour and existing `test_los.py` golden traces stay green.
**Plans**: 2 plans

Plans:
- [ ] 01-01-PLAN.md — Terrain config schema (`TerrainPieceConfig` + `WargameEnvConfig.terrain`) with bounds/non-overlap validation, pure `domain/terrain.py` (`Footprint`/`Terrain`), domain + config tests; Wave-0 hypothesis dep
- [ ] 01-02-PLAN.md — Wire `battle_factory`→`Battle.terrain`→`BattleView.terrain`; endpoint-aware symmetric LOS seam; renderer footprint overlay + verdict-coloured LOS; golden boards + symmetry/consistency/movement tests; demo `terrain_los_demo.yaml`

### Phase 2: Terrain in the Observation
**Goal**: Terrain footprints are encoded in the agent's observation as an entity-token
stream appended last, so the policy can reason about it without any mid-episode shape change and
without breaking pre-terrain checkpoints.
**Directory**: `02-terrain-observation`
**Depends on**: Phase 1 (requires `BattleView.terrain`)
**Requirements**: TERR-08, TERR-09
**Success Criteria** (what must be TRUE):
  1. With terrain configured, the observation includes terrain tokens carrying geometry only (footprint corners/bbox, normalised) — one token per footprint, token count equals footprint count, and no per-enemy visibility side-channel leaks privileged info.
  2. With no terrain configured, the observation adds nothing: tensor shapes and dtypes are byte-identical to a captured pre-terrain golden, and there is no mid-episode observation shape change.
  3. A pre-terrain checkpoint loads and infers on a no-terrain config and produces numerically-equal logits/values, because `terrain_embedding` is `None` when `terrain_size == 0` and terrain tokens are appended after opponents.
  4. Player and opponent token positions, per-model action heads, and the critic token are unchanged when terrain is present, across the Transformer, MLP, and PPO networks.
**Plans**: TBD

Plans:
- [ ] 02-01: TBD (`WargameTerrainObservation` + `env_observation.terrain` + builder `_terrain_to_obs`; terrain tensor in `model/common/observation.py`, None-guarded, inserted before mask)
- [ ] 02-02: TBD (`terrain_embedding` appended LAST in Transformer + MLP + PPO backbone share; `from_env` reads `terrain_size`; backward-compat tests: no-terrain byte-identical + pre-terrain checkpoint loads & infers)

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2
(Phase 2 depends on `BattleView.terrain` from Phase 1 — strictly sequential)

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Terrain in the Simulation (LOS-Blocking) | 0/2 | Not started | - |
| 2. Terrain in the Observation | 0/2 | Not started | - |
