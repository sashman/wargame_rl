# Requirements: Wargame RL — Terrain & Line-of-Sight Blocking (v2.0)

**Defined:** 2026-06-19
**Core Value:** Agents learn recognisable tactical behaviour through reward shaping and environment design

> Prior milestones (v1.0, v9.0) are complete. Details in git history.

## Milestone Scope

Terrain (Ruins) that blocks **line of sight only**, using the canonical Warhammer 40k 10e **Ruins
abstraction**: a terrain piece is a **footprint rectangle** and the *footprint itself* is the LOS
blocker. A ruin blocks the line between two models only when its footprint lies between them and
**both** models are outside it; a model inside the footprint can **see out** and be **seen into**
(see-into / see-out exceptions). Walls have **no LOS role** this milestone and are deferred. Movement
is unaffected. Terrain is encoded in the observation so the agent can reason about it. The
footprint-based blocking predicate plugs into the existing single LOS seam (`domain/los.py`
Bresenham unchanged). See `01-terrain-los-blocking/01-CONTEXT.md` and the 10e Ruins rules cited
there; note the research `SUMMARY.md`'s wall-rasterisation recommendation is **superseded** by this
footprint-based decision.

## v2.0 Requirements

Each requirement maps to exactly one roadmap phase.

### Terrain Model & Configuration

- [ ] **TERR-01**: A terrain piece (ruin) is configurable in YAML as a footprint rectangle, given as two opposite corners `[x0, y0, x1, y1]` in absolute board coordinates
- [ ] **TERR-02**: Terrain config is validated at load — footprint corners fall within board bounds and footprints do not overlap each other
- [ ] **TERR-03**: Configs with no terrain behave exactly as today; existing YAML configs and pre-terrain checkpoints keep working (no-op default)

### Line of Sight

- [ ] **TERR-04**: A ruin footprint blocks line of sight when it lies between two models that are **both outside** it; a model inside a footprint can see out of it and be seen into (10e see-out / see-into exceptions, evaluated per ruin)
- [ ] **TERR-05**: Terrain does not affect movement — models traverse and may occupy footprint cells freely
- [ ] **TERR-06**: Footprint LOS blocking flows through the single LOS service (an endpoint-aware blocking predicate) so shooting masks, action masks, resolution, and rendering all agree on the same blocking
- [ ] **TERR-07**: LOS is symmetric (A sees B iff B sees A) and deterministic on known board configurations

### Observation

- [ ] **TERR-08**: Terrain footprints are encoded in the agent's observation so the policy can reason about them
- [ ] **TERR-09**: Terrain encoding causes no mid-episode observation shape change and adds nothing to the observation when no terrain is configured

### Rendering

- [ ] **TERR-10**: The renderer draws terrain footprints (translucent fill + outline + label), and the LOS debug overlay is coloured by the actual blocked/clear verdict

## Future Requirements

Deferred to a later terrain milestone (tracked, not in this roadmap):

- **TERRF-01**: Walls within a footprint (thin L-shaped segments) — for rendering realism, movement interaction, and/or finer-grained LOS than the footprint abstraction
- **TERRF-02**: Dense/Woods-style visibility ("not fully visible" partial obscuring) distinct from the Ruins footprint block
- **TERRF-03**: Cover bonus / Benefit of Cover (+1 armour save vs ranged when not fully visible due to terrain) — maps onto the footprint model
- **TERRF-04**: Difficult terrain (movement speed penalty)
- **TERRF-05**: Impassable/blocking terrain (models cannot move through)
- **TERRF-06**: Elevation and height advantage (improved AP from elevated positions)
- **TERRF-07**: Board templates and procedural terrain placement for training variety
- **TERRF-08**: Variable base sizes (research spike — impact on grid, coherency, engagement, LOS)
- **TERRF-09**: Terrain in `GameStateSnapshot` for cross-consumer / LLM / replay completeness

## Out of Scope

| Feature | Reason |
|---------|--------|
| Walls as LOS blockers this milestone | 10e abstracts ruin LOS to the footprint (can't see through even via windows/doors); walls deferred to a later milestone for rendering/movement/finer LOS |
| Footprint gameplay effects beyond LOS (cover, dense visibility, difficult ground) | This milestone is LOS-blocking only; cover/visibility effects deferred to a later terrain milestone |
| Terrain affecting movement (impassable, difficult) | Milestone is LOS-only; movement untouched |
| Elevation / 3D / height | Discrete 2D grid only this milestone |
| Procedural / template terrain generation | Hand-authored YAML terrain first; generation deferred |
| Variable base sizes | Research spike; large cross-cutting change deferred |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| TERR-01 | Phase 1 | Pending |
| TERR-02 | Phase 1 | Pending |
| TERR-03 | Phase 1 | Pending |
| TERR-04 | Phase 1 | Pending |
| TERR-05 | Phase 1 | Pending |
| TERR-06 | Phase 1 | Pending |
| TERR-07 | Phase 1 | Pending |
| TERR-08 | Phase 2 | Pending |
| TERR-09 | Phase 2 | Pending |
| TERR-10 | Phase 1 | Pending |

*Phase column filled by the roadmapper. TERR-03 (no-op default / backward compat) is assigned to
Phase 1 for traceability; its observation-side guarantee (no-terrain adds nothing, pre-terrain
checkpoints load) is also verified in Phase 2 via TERR-09's backward-compat tests.*

---
*Requirements defined: 2026-06-19 — v2.0 Terrain & Line-of-Sight Blocking*
