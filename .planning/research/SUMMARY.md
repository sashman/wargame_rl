# Project Research Summary

**Project:** wargame_rl — v2.0 Terrain & Line-of-Sight Blocking
**Domain:** Brownfield discrete-grid RL tabletop wargame (DDD env, transformer-over-entities observation)
**Researched:** 2026-06-19
**Confidence:** HIGH

## Executive Summary

This milestone adds **terrain that blocks line of sight only** to an existing, working PPO/DQN wargame env. A terrain piece is a **footprint rectangle** (an area marker with *no* gameplay effect this milestone) plus **thin L-shaped walls** that block LOS but leave movement untouched, with all terrain **encoded into the observation**. The four research tracks reached an unusually clean consensus: this is a *pure domain-logic + NumPy + existing-Pydantic-config* change — **no new dependencies** — that extends surfaces the codebase already has (the single Bresenham LOS seam, the entity-token transformer, the config→factory→aggregate→view DDD spine).

The recommended approach is **whole-cell wall blocking**: author walls as thin segments, but **rasterise them to whole blocking cells** in a new pure `domain/terrain.py` service, OR them into the existing injectable `is_blocking(x, y)` predicate, and leave `domain/los.py`'s Bresenham core completely unchanged. The footprint is deliberately **not** rasterised (no LOS effect yet). Terrain is encoded as a **new entity-token stream appended LAST** in the observation, with a `terrain_embedding` that is `None` when no terrain is present — exactly mirroring the proven `opponent_model_embedding is None` pattern. This yields the central backward-compat guarantee: **no terrain ⇒ zero terrain tokens ⇒ byte-identical behaviour and pre-terrain checkpoints still load.** Cell-edge "true thin" walls are explicitly deferred (they would require new ray traversal and break the tested interior-only contract).

The work splits into a natural **two-phase build order** with a clean one-way dependency: **Phase A** puts terrain into the simulation/LOS (config + rasteriser + Battle/BattleView wiring + repoint `_make_is_blocking` + renderer + tests); **Phase B** puts terrain into the observation (obs type + tensor pipeline + both networks + backward-compat tests). The top risks are all geometry/contract integrity issues — rasterised L-wall corner pinholes or sealed gaps, Bresenham `A→B` vs `B→A` asymmetry, mask↔resolution LOS divergence, observation shape drift / checkpoint break, observation honesty, and renderer/debug divergence — every one of which has a concrete, testable mitigation. The two non-negotiable tests are a **Hypothesis LOS-symmetry property test** and **golden L-wall boards**.

## Key Findings

### Recommended Stack

No stack change. Everything is built on libraries already pinned in the project; the integration surfaces (LOS seam, blocking predicate, config, observation pipeline, transformer) already exist and are the only things touched. See `STACK.md`.

**Core technologies:**
- **NumPy** (already pinned): wall rasterisation to a boolean blocking grid + vectorised observation features — grid blocking is a 2D bool array; no geometry lib needed.
- **Pydantic + pydantic-yaml** (already pinned): new `TerrainPieceConfig` / `WallConfig` models with init-time validation, mirroring the existing `blocking_mask` / `ObjectiveConfig` pattern.
- **In-repo Bresenham** (`domain/los.py`): single LOS source of truth, reused unchanged; walls feed it via the existing `is_blocking` injection point.
- **Explicitly NOT added:** Shapely (GEOS native dep, continuous geometry unused on a grid, slower than integer Bresenham), SciPy/`ndimage` (only relevant to deferred dense-visibility), roguelike FOV libs like `tcod`. No new dependency is justified.

### Expected Features

Scope is *line of sight only*. See `FEATURES.md`.

**Must have (table stakes):**
- **Terrain config schema** — `TerrainPieceConfig` (footprint rect + `walls`), validated at construction (in-bounds, walls ⊂ footprint), default no-op (`None`).
- **Walls compiled into the single LOS blocking predicate** — whole-cell rasterisation OR'd with the existing `blocking_mask`; one Bresenham path for masks, resolution, renderer, snapshot.
- **Terrain encoded in the observation** — terrain tokens through the full obs pipeline + **both** networks, fixed token semantics, no mid-episode shape change.
- **Backward-compatible no-op default** — absent/empty terrain reproduces today's tensor shapes and LOS behaviour exactly.
- **Renderer draws footprints + walls** — drawn from the rasterised grid so the debug `L` LOS overlay agrees with what actually blocks.
- **Tests** — wall-blocks-ray, open-lane-passes, no-terrain regression, obs-shape stability (deterministic).

**Should have (cheap polish only):**
- Footprint-membership obs flag (cheap precursor to future cover).
- Terrain in `GameStateSnapshot` (cross-consumer/LLM/replay completeness).
- Multiple terrain pieces per board (falls out of `list[TerrainPieceConfig]`).

**Defer (later terrain milestone):**
- Cell-edge "true thin" walls, footprint-obscures-LOS (40k-faithful ruin), cover save bonus, dense visibility, difficult/impassable ground, elevation, board templates/procedural placement, variable base sizes, per-cell full-grid terrain channel.

> **Design divergence to flag:** real 40k 10e has the *footprint* obscure LOS; this milestone intentionally inverts that — **walls block, footprint is a no-op marker**. Requirements should state "walls block the ray," not "footprint obscures the area."

### Architecture Approach

Terrain slots into the existing DDD spine (config → `battle_factory` → `Battle` aggregate → `BattleView` → consumers) identically to how objectives already do. The terrain blocking grid is **precomputed once at `Battle` construction** (terrain is static per episode), removing the current per-query closure rebuild, and exposed behind the single `has_line_of_sight_between_cells` seam so **every consumer (shooting masks, action masks, renderer, snapshot) gets terrain for free with zero call-site changes.** See `ARCHITECTURE.md`.

**Major components:**
1. **`domain/terrain.py`** (NEW, pure) — `Footprint`, `Wall`, `TerrainPiece` value objects + `build_los_blocking_grid(terrain, base_mask, w, h)` rasteriser (walls only; footprint NOT rasterised).
2. **`Battle` / `BattleView`** (MOD) — hold `terrain` + precomputed `los_blocking_grid`; read-only `terrain` accessor for obs/render.
3. **`wargame.py::_make_is_blocking`** (MOD) — the single behavioural edit for blocking; sources the merged grid; `domain/los.py` stays untouched.
4. **Observation pipeline** (MOD) — `WargameTerrainObservation` → `env_observation.terrain` → `observation_builder._terrain_to_obs` → terrain tensor (inserted before mask: `[game, obj, players, opp, terrain, mask]`) → `terrain_embedding` (None-guarded) appended LAST in **both** networks + PPO.
5. **`renders/human.py`** (MOD) — draw footprints + walls from the rasterised grid; colour debug LOS by actual verdict.

### Critical Pitfalls

Top risks (all from `PITFALLS.md`), each with a testable mitigation:

1. **Rasterised L-wall corner pinholes / sealed gaps** — Bresenham wall cells are diagonally (not 4-)connected, so rays slip through elbows, or over-thickening seals an intended lane. Mitigate: guarantee 4-connectivity in the rasteriser (not in the LOS trace), author the corner cell explicitly, and add golden L-wall boards testing diagonal-through-elbow and intended-gap rays. *(Phase A)*
2. **LOS symmetry break (`A→B` clear, `B→A` blocked)** — Bresenham is direction-dependent near cell corners; RL will exploit one-way firing lanes. Mitigate: **Hypothesis property test `los(A,B)==los(B,A)`** (highest-value test in the milestone); if it fails, canonicalise endpoint order in the seam. *(Phase A)*
3. **Mask ↔ resolution LOS divergence** — if the shooting mask and resolution use different predicates, the agent is offered illegal shots / makes masked-out ones. Mitigate: one precomputed grid, all consumers route through `has_line_of_sight_between_cells`; consistency test that masked pairs == LOS-clear ∩ range ∩ alive. *(Phase A)*
4. **Observation shape drift / checkpoint break** — terrain inserted mid-list or `terrain_embedding` instantiated at `terrain_size==0` shifts indices or state-dict keys. Mitigate: `None` embedding when absent, append terrain LAST, fix unpackers once (`opp=xs[3]`, `terrain=xs[4]`, `mask=xs[5]`); backward-compat tests including **pre-terrain checkpoint loads & infers**; verify Transformer **and** MLP **and** PPO. *(Phase B)*
5. **Renderer / debug LOS divergence + observation honesty** — drawing authored thin segments while LOS blocks whole cells creates visual/logical mismatch; terrain tokens must carry geometry only (no per-enemy visibility side-channel) and round-trip to the blocking cells. Mitigate: render from the rasterised grid, colour debug line by verdict; geometry-only tokens, token count == wall count. *(Phase A render; Phase B honesty)*

Also flagged (curriculum/eval concern, out of strict milestone scope): **reward exploitation via wall-camping / "hide-and-plink"** and **distribution shift when terrain is switched on** — keep a sparse objective anchor and treat terrain-enabled as a distinct training/curriculum config, not an in-place flag flip.

## Implications for Roadmap

Research strongly converges on a **two-phase structure** with a clean one-way dependency (B needs `BattleView.terrain` from A). A is *mechanics* (does the rule work?), B is *perception* (can the agent see it?) — separate test surfaces, separate risk profiles, each backward-compat guarantee independently verifiable.

### Phase A: Terrain in the simulation (LOS-blocking)
**Rationale:** Unlocks everything and depends on nothing new; isolates the geometry/LOS-seam risk from the tensor/network risk.
**Delivers:** Config types + validators (`TerrainPieceConfig`, `WallConfig`, `WargameEnvConfig.terrain` default `None`); pure `domain/terrain.py` value objects + `build_los_blocking_grid`; `battle_factory` → `Battle.terrain` + precomputed `los_blocking_grid`; `BattleView.terrain`; repointed `_make_is_blocking`; renderer overlay drawn from the rasterised grid.
**Addresses:** Terrain config schema, walls-block-LOS (whole-cell), backward-compatible no-op default, renderer footprints+walls.
**Avoids:** Pitfalls 1 (corner pinhole/seal), 2 (symmetry), 3 (interior-only contract), 4 (mask↔resolution), 8 (renderer divergence).
**Exit:** shooting/action masks and `has_line_of_sight_between_cells` respect walls; movement unchanged; existing tests green; LOS-symmetry property test + golden L-wall boards pass.

### Phase B: Terrain in the observation
**Rationale:** Depends on Phase A (`BattleView.terrain`); owns the tensor/network contract and checkpoint compatibility.
**Delivers:** `WargameTerrainObservation` + `env_observation.terrain` + `_terrain_to_obs`; terrain tensor in `model/common/observation.py` (insert before mask, fix both converters); `terrain_embedding` (None-guarded) appended LAST in **both** networks + PPO backbone share; `from_env` reads `terrain_size`; an example terrain config + a no-terrain config.
**Uses:** NumPy feature encoding + the entity-token transformer (`net.py`), the `opponent_model_embedding is None` precedent.
**Implements:** The observation-pipeline component end to end.
**Avoids:** Pitfalls 5 (shape drift/checkpoint break), 9 (observation honesty), 7 (distribution shift — ship as a distinct config/curriculum stage).
**Exit:** agent observes terrain; no-terrain runs reproduce prior behaviour byte-identically; a pre-terrain checkpoint still loads and infers; player/opponent token positions unchanged.

### Phase Ordering Rationale
- **Dependency-driven:** B literally cannot start without `BattleView.terrain` from A; A introduces no new dependency on B.
- **Risk isolation:** A's risk is domain geometry + the LOS seam; B's risk is the tensor/network/checkpoint contract. Splitting keeps each PR small and each backward-compat guarantee independently testable.
- **Avoids the cross-cutting pitfalls** by construction: single blocking seam (no mask/resolution drift), append-last token stream (no index drift), rasterise-walls-only (footprint never blocks).

### Research Flags
Phases likely needing deeper research/decision confirmation during planning:
- **Phase A:** Confirm the **4-connectivity rasterisation rule** for L-walls (supercover vs orthogonal-filler at diagonal steps) and the **endpoint-canonicalisation rule** for LOS symmetry. These are the only genuinely open design choices; both are pure-Python, well-bounded, and testable. (The whole-cell-vs-cell-edge decision itself is already settled: whole-cell, cell-edge deferred.)

Phases with standard patterns (skip deeper research):
- **Phase B:** Adding an entity-token stream is a fully documented, precedented procedure (the `pytorch-dqn.mdc` "add a new entity" checklist + the opponents `None`-embedding template). Mechanical, not novel.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | "No new deps" verified against `pyproject.toml` + existing seams; reuses v1 conclusion. |
| Features | HIGH | Grounded in code + `PROJECT.md` + tabletop rules ref; MEDIUM only on the (now-settled) wall-representation choice. |
| Architecture | HIGH | Grounded in a full read of the env source (`los.py`, `battle.py`, `battle_factory.py`, `battle_view.py`, `net.py`, etc.). |
| Pitfalls | HIGH | Codebase-specific pitfalls grounded in source + `test_los.py`; MEDIUM only for RL training/exploitation patterns (literature + community consensus). |

**Overall confidence:** HIGH

### Gaps to Address
- **Rasteriser connectivity rule (4-connected walls):** decide and document in Phase A planning; verify with golden L-wall boards (diagonal elbow + intended gap). Recovery cost is low if caught before training.
- **LOS symmetry canonicalisation:** the Hypothesis property test may fail on some boards; plan the canonical endpoint ordering as the seam-level fix up front. Low recovery cost pre-training.
- **Whole-cell accuracy loss (MEDIUM):** acceptable this milestone (movement unaffected, grid already bounds LOS precision, cover deferred); revisit as a dedicated supercover-LOS phase only if a future milestone needs true thin walls.
- **Curriculum / wall-camping (out of strict scope):** flag for the combat-reward / curriculum phase — keep a sparse objective anchor and a pressuring opponent; ship terrain as a distinct config, not an in-place flag flip.

## Sources

### Primary (HIGH confidence)
- In-repo code — `envs/domain/los.py`, `envs/wargame.py` (`_make_is_blocking`, `has_line_of_sight_between_cells`), `envs/types/config.py` (`blocking_mask`), `domain/battle.py` / `battle_factory.py` / `battle_view.py`, `env_components/observation_builder.py` + `shooting_masks.py`, `types/env_observation.py`, `model/common/observation.py`, `model/net.py`, `renders/human.py`, `tests/test_los.py`, `pyproject.toml`.
- `.cursor/rules/pytorch-dqn.mdc`, `gymnasium-env.mdc` — "add a new entity" obs/tensor/network checklist.
- `docs/ddd-envs.md` — domain/BattleView extension boundaries; `docs/tabletop-rules-reference.md` — terrain/ruins LOS behaviour.
- `.planning/PROJECT.md` — milestone scope, deferral list, backward-compat constraint.
- Prior research — `.planning/research/v1-STACK.md`, `v1-FEATURES.md`, `v1-PITFALLS.md`.

### Secondary (MEDIUM confidence)
- Warhammer Community "Simple Terrain", Goonhammer Ruleshammer Terrain Guide, tuser.tv 10e ruins/visibility — current 40k 10e footprint/wall LOS (used to flag the footprint-vs-wall divergence).
- Bresenham diagonal-connectivity / supercover / direction-dependence (Wikipedia + standard grid-LOS/FOV references).
- RL reward-shaping & distribution-shift patterns — strategy-game RL community consensus.

### Tertiary (LOW confidence)
- None — all findings trace to code, project docs, or multiple agreeing external sources.

---
*Research completed: 2026-06-19*
*Ready for roadmap: yes*
