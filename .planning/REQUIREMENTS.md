# Requirements: Wargame RL — Terrain & Line-of-Sight Blocking (v2.0)

**Defined:** 2026-06-19
**Core Value:** Agents learn recognisable tactical behaviour through reward shaping and environment design

> Prior milestones are archived under `.planning/milestones/`:
> v1.0 (Shooting & Model Destruction) → `v1-REQUIREMENTS.md` / `v1-ROADMAP.md` (Phase 6 deferred);
> v9.0 (Structured Game State) → `v9-REQUIREMENTS.md` / `v9-ROADMAP.md` (shipped).

## Milestone Scope

Terrain that blocks **line of sight only**, modeled as a **footprint** (a large rectangle area
marker with no gameplay effect this milestone) plus thin **walls** (segments within the footprint,
usually L-shaped) that block LOS. Movement is unaffected. Terrain is encoded in the observation so
the agent can reason about it. Wall grid representation is **whole-cell rasterisation** behind the
existing single LOS seam (cell-edge "true thin" walls deferred). See `.planning/research/SUMMARY.md`.

## v2.0 Requirements

Each requirement maps to exactly one roadmap phase.

### Terrain Model & Configuration

- [ ] **TERR-01**: A terrain piece is configurable in YAML as a footprint rectangle plus zero or more thin walls (e.g. L-shaped) placed within that footprint
- [ ] **TERR-02**: Terrain config is validated at load — walls lie within their footprint and all coordinates fall within board bounds
- [ ] **TERR-03**: Configs with no terrain behave exactly as today; existing YAML configs and pre-terrain checkpoints keep working (no-op default)

### Line of Sight

- [ ] **TERR-04**: Walls block line of sight — a shot whose ray crosses a wall cell is blocked; footprints do not block LOS
- [ ] **TERR-05**: Walls and footprints do not affect movement — models traverse terrain freely
- [ ] **TERR-06**: Wall LOS blocking flows through the single LOS service so shooting masks, action masks, resolution, and rendering all agree on the same blocking
- [ ] **TERR-07**: LOS through walls is symmetric (A sees B iff B sees A) and deterministic on known board configurations

### Observation

- [ ] **TERR-08**: Terrain (footprints and walls) is encoded in the agent's observation so the policy can reason about it
- [ ] **TERR-09**: Terrain encoding causes no mid-episode observation shape change and adds nothing to the observation when no terrain is configured

### Rendering

- [ ] **TERR-10**: The renderer draws terrain footprints and walls, and the LOS debug overlay reflects actual wall blocking

## Future Requirements

Deferred to a later terrain milestone (tracked, not in this roadmap):

- **TERRF-01**: Cell-edge "true thin" walls (sub-cell LOS via supercover traversal)
- **TERRF-02**: Footprint-obscures-LOS (40k-faithful ruin obscuring, dense visibility)
- **TERRF-03**: Cover bonus (+1 armour save vs ranged when in/behind cover)
- **TERRF-04**: Difficult terrain (movement speed penalty)
- **TERRF-05**: Impassable/blocking terrain (models cannot move through)
- **TERRF-06**: Elevation and height advantage (improved AP from elevated positions)
- **TERRF-07**: Board templates and procedural terrain placement for training variety
- **TERRF-08**: Variable base sizes (research spike — impact on grid, coherency, engagement, LOS)
- **TERRF-09**: Terrain in `GameStateSnapshot` for cross-consumer / LLM / replay completeness

## Out of Scope

| Feature | Reason |
|---------|--------|
| Cell-edge thin walls this milestone | Whole-cell rasterisation reuses the existing Bresenham seam with no LOS-core change; cell-edge needs new ray traversal — deferred |
| Footprint gameplay effects (cover, dense visibility, difficult ground) | Footprint is a no-op area marker this milestone; effects deferred to a later terrain milestone |
| Terrain affecting movement (impassable, difficult) | Milestone is LOS-only; movement untouched |
| Elevation / 3D / height | Discrete 2D grid only this milestone |
| Procedural / template terrain generation | Hand-authored YAML terrain first; generation deferred |
| Variable base sizes | Research spike; large cross-cutting change deferred |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| TERR-01 | TBD | Pending |
| TERR-02 | TBD | Pending |
| TERR-03 | TBD | Pending |
| TERR-04 | TBD | Pending |
| TERR-05 | TBD | Pending |
| TERR-06 | TBD | Pending |
| TERR-07 | TBD | Pending |
| TERR-08 | TBD | Pending |
| TERR-09 | TBD | Pending |
| TERR-10 | TBD | Pending |

*Phase column filled by the roadmapper.*

---
*Requirements defined: 2026-06-19 — v2.0 Terrain & Line-of-Sight Blocking*
