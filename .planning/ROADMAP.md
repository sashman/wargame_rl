# Roadmap: Terrain & Line-of-Sight Blocking (v2.0)

## Milestones

- ✅ **v9.0 Structured Game State & LLM-Readable Representation** — shipped 2026-06-19 ([archive](milestones/v9-ROADMAP.md))
- ✅ **v1.0 Shooting & Model Destruction** — Phases 1–5 complete; Phase 6 (Combat Reward & Curriculum) deferred ([archive](milestones/v1-ROADMAP.md))
- 🚧 **v2.0 Terrain & Line-of-Sight Blocking** — Phases 1–2 (this roadmap)

## Overview

This milestone adds terrain that blocks **line of sight only** to the existing PPO/DQN wargame
env. A terrain piece is a **footprint rectangle** (an area marker with no gameplay effect this
milestone) plus thin **walls** (usually L-shaped segments inside the footprint) that block LOS while
leaving movement untouched, with all terrain **encoded in the observation**. Research reached a clean
consensus: this is a pure domain-logic + NumPy + existing-Pydantic-config change with **no new
dependencies**, extending surfaces the codebase already has (the single Bresenham LOS seam, the
entity-token transformer, the config→factory→aggregate→view DDD spine).

The work splits into a natural **two-phase build order** with a clean one-way dependency: Phase 1
puts terrain into the simulation/LOS (does the rule work?); Phase 2 puts terrain into the observation
(can the agent see it?). Phase 2 needs `BattleView.terrain` from Phase 1. Walls are authored as thin
segments but **rasterised to whole blocking cells** OR'd into the existing `is_blocking(x, y)` seam —
the Bresenham core in `domain/los.py` stays untouched, and the footprint is deliberately not
rasterised (no LOS effect yet). The central backward-compat guarantee: **no terrain ⇒ zero terrain
tokens ⇒ byte-identical behaviour and pre-terrain checkpoints still load.**

> **Phase directory naming:** This milestone restarts phase numbering at 1 with unprefixed
> directories (`01-terrain-los-blocking`, `02-terrain-observation`). The completed v1.0 phase
> directories were archived to `.planning/milestones/v1.0-phases/` (and the v9.0 phase dir to
> `.planning/milestones/v9.0-phases/`) so the `01-`/`02-` namespace is free and the GSD tooling
> resolves these phases correctly.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Terrain in the Simulation (LOS-Blocking)** - Config + domain rasteriser + LOS seam + renderer so walls block line of sight while movement is unaffected
- [ ] **Phase 2: Terrain in the Observation** - Terrain entity-token stream through the obs pipeline and both networks so the agent can see and reason about terrain

## Phase Details

### Phase 1: Terrain in the Simulation (LOS-Blocking)
**Goal**: Walls authored in YAML block line of sight through the single LOS service while movement
passes through freely; configs with no terrain behave exactly as today.
**Directory**: `01-terrain-los-blocking`
**Depends on**: Nothing (first phase; depends only on the existing domain/LOS layer)
**Requirements**: TERR-01, TERR-02, TERR-03, TERR-04, TERR-05, TERR-06, TERR-07, TERR-10
**Success Criteria** (what must be TRUE):
  1. A YAML config can declare a terrain piece (footprint rectangle + zero or more thin L-shaped walls); loading is rejected with a clear error when a wall lies outside its footprint or any coordinate falls off the board.
  2. A shot whose ray crosses a wall cell is blocked, an open lane stays clear, and a footprint alone never blocks LOS — verified by golden L-wall boards including the diagonal elbow and an intended 1-cell gap.
  3. LOS through walls is symmetric (A sees B iff B sees A) and deterministic on known boards — verified by a Hypothesis property test over randomised grids and endpoints.
  4. Shooting masks, action masks, shooting resolution, the renderer overlay, and the snapshot all agree on the same wall blocking because they route through the single `has_line_of_sight_between_cells` seam; the renderer draws walls from the rasterised blocking grid and colours the debug LOS line by the actual verdict.
  5. A model can still move through a wall cell (movement unaffected), and an existing no-terrain config produces byte-identical LOS behaviour and existing `test_los.py` golden traces stay green.
**Plans**: TBD

Plans:
- [ ] 01-01: TBD (config types + validators; `domain/terrain.py` value objects + `build_los_blocking_grid` rasteriser, unit-tested without Gym)
- [ ] 01-02: TBD (wire `battle_factory` → `Battle.terrain` + precomputed `los_blocking_grid`; `BattleView.terrain`; repoint `_make_is_blocking`; renderer overlay; LOS-symmetry property test + golden L-wall boards)

### Phase 2: Terrain in the Observation
**Goal**: Terrain (footprints and walls) is encoded in the agent's observation as an entity-token
stream appended last, so the policy can reason about it without any mid-episode shape change and
without breaking pre-terrain checkpoints.
**Directory**: `02-terrain-observation`
**Depends on**: Phase 1 (requires `BattleView.terrain`)
**Requirements**: TERR-08, TERR-09
**Success Criteria** (what must be TRUE):
  1. With terrain configured, the observation includes terrain tokens carrying geometry only (wall endpoints + footprint bbox, normalised) — one token per wall, token count equals wall count, and no per-enemy visibility side-channel leaks privileged info.
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
