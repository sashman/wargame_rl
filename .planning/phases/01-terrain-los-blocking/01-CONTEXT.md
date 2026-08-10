# Phase 1: Terrain in the Simulation (LOS-Blocking) - Context

**Gathered:** 2026-06-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Add terrain pieces (Ruins) that block **line of sight** in the simulation, using the canonical
**Ruins** abstraction: LOS is determined by the terrain **footprint** (a 2D
rectangle on the grid), not by wall geometry. Movement is unaffected. Configs with no terrain
behave exactly as today. (Encoding terrain into the agent observation is Phase 2.)

**In scope:** footprint config schema + validation, footprint-based LOS through the single LOS
seam, renderer overlay, tests (LOS symmetry + golden boards), backward-compatible no-op default.

**Out of scope this milestone:** walls (deferred — revisit later for rendering/movement/finer LOS),
cover, dense visibility, difficult/impassable movement, elevation/Plunging Fire, board templates,
variable base sizes, terrain in the observation (Phase 2).
</domain>

<decisions>
## Implementation Decisions

### LOS model — canonical Ruins (footprint-based)
- **D-01:** The **footprint** is the LOS primitive. A ruin blocks line of sight between observer O
  and target T **iff** the line O→T passes through that ruin's footprint **AND both O and T are
  outside that footprint**.
- **D-02:** **See-into exception** — if T is inside the footprint, T is visible to O (you can see
  *into* a ruin normally).
- **D-03:** **See-out exception** — if O is inside the footprint, O can see out normally. (Two
  models inside the same footprint always see each other.)
- **D-04:** The exception is **per-ruin**: being inside ruin A's footprint does **not** let you see
  through a *different* ruin B that lies between O and T. Each ruin is evaluated independently.
- **D-05:** **Walls play NO role in LOS this milestone.** The reference rules abstract LOS to the footprint — you
  cannot see through a ruin even via windows/doors/open sides. Walls are deferred (see Deferred).
- **D-06:** Movement is unaffected — models may move through and occupy footprint cells freely.
- **D-07:** Implication for the LOS seam: blocking is **endpoint-aware** (depends on which
  footprints O and T sit in), so the existing injectable `is_blocking(x, y)` predicate must be
  built **per query** from the endpoints (a cell is blocking iff it is inside some footprint F with
  O∉F and T∉F). The `domain/los.py` Bresenham core stays unchanged. This generalises (does not
  replace) the existing `blocking_mask`, which keeps working as an always-on blocking grid.

### Config schema
- **D-08:** A terrain piece (ruin) footprint is written as two opposite corners
  `[x0, y0, x1, y1]` in **absolute** board coordinates (consistent with models/objectives/
  `deployment_zone`/`blocking_mask`). No wall config this milestone.
- **D-09:** Terrain config defaults to **None/empty** (no-op). Existing YAML and pre-terrain
  checkpoints keep working unchanged.

### Placement & validation
- **D-10:** Footprint corners must be validated **within board bounds** at load (clear error
  otherwise).
- **D-11:** Footprints **may overlap** deployment zones and objectives, and models **may spawn on /
  occupy** footprint cells (consistent with the see-in/see-out rule).
- **D-12:** Footprints must **not overlap each other** (keep pieces distinct) — validated at load.

### Rendering
- **D-13:** Draw each footprint as a **translucent filled rectangle with an outline + label**.
- **D-14:** Colour the debug LOS line by the actual **blocked/clear verdict** (so the human overlay
  agrees with the domain LOS result).

### Demo scenario
- **D-15:** Ship a **new example env config**: two facing deployment zones with one or two ruins
  between them that block cross-board sightlines (a clear LOS demonstration).

### Claude's Discretion
- Exact footprint rasterisation / "passes through footprint" cell test (which interior cells count),
  endpoint canonicalisation for LOS symmetry, and the precise rendering colours/alpha — planner/
  implementer decides, constrained by the tests below.
- Whether footprints are stored as rectangles + a derived membership test vs a precomputed boolean
  footprint grid — implementer's call (performance/clarity).

### Required tests (carried from research)
- **LOS symmetry property test:** `los(A, B) == los(B, A)` over randomised boards/endpoints
  (footprint-based LOS should be naturally symmetric — assert it).
- **Golden boards:** ruin between two outside models → blocked; observer or target inside footprint
  → visible (see-out / see-in); ruin not on the line → unaffected; no-terrain config → byte-identical
  to today.
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Rules reference (authoritative for the LOS model)
- `docs/rules/13-terrain.md` — terrain categories, the solid rule and the see-out / see-into
  behaviour that D-01..D-06 derive from: sight cannot be drawn through an enclosed ground-level
  gap, but a model inside a feature can see out of it and be seen into.
- `docs/rules/06-visibility-and-damage.md` — line of sight, *visible* vs *fully visible*.
- `docs/rules/constants.yaml` — footprint and battlefield dimensions for the demo scenario
  scale (D-15).

### Project research (this milestone)
- Prior research docs (ARCHITECTURE, PITFALLS, SUMMARY) removed during planning compaction.
  Key takeaways preserved in this CONTEXT and the plans. Wall-rasterisation recommendation is
  **superseded** by the footprint-based LOS decision here.

### Codebase
- `wargame_rl/wargame/envs/domain/los.py` — Bresenham + injectable `is_blocking` seam (unchanged).
- `wargame_rl/wargame/envs/types/config.py` — `WargameEnvConfig`, `blocking_mask`, `deployment_zone`,
  `ObjectiveConfig` (schema/validator patterns to mirror).
- `wargame_rl/wargame/envs/wargame.py` — `_make_is_blocking`, `has_line_of_sight_between_cells` seam.
- `docs/rules/13-terrain.md` — project's terrain rules reference.
- `.cursor/rules/gymnasium-env.mdc` — "Adding Features" checklist (config → env_components → obs →
  tensor → networks → renderer → tests + backward compat).
</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `domain/los.py::has_line_of_sight(..., is_blocking)` — single LOS source of truth; takes an
  injectable blocking predicate. The footprint-based, endpoint-aware predicate plugs straight in.
- `WargameEnvConfig.blocking_mask` + its `field_validator`/`model_validator` — template for the new
  terrain config field and bounds validation.
- `deployment_zone: tuple[int,int,int,int]` — precedent for a rectangle in config (corner/size tuple).

### Established Patterns
- DDD spine: config → `battle_factory` → `Battle` aggregate → `BattleView` → consumers (objectives
  follow this; terrain mirrors it).
- New config fields default to no-op for backward compatibility (e.g. `blocking_mask=None`,
  `number_of_opponent_models=0`).
- `_make_is_blocking` already rebuilds a blocking closure — extend it to be endpoint-aware.

### Integration Points
- `_make_is_blocking` / `has_line_of_sight_between_cells` — single behavioural edit; shooting masks,
  action masks, resolution, renderer, and snapshot all route through this seam, so they inherit
  terrain LOS for free.
- Renderer (`renders/human.py`) — add footprint overlay + verdict-coloured debug LOS line.
</code_context>

<specifics>
## Specific Ideas

- The LOS interaction must follow the Ruins visibility rule exactly (see canonical refs). The
  user explicitly reset to the source rules to avoid layered assumptions — do not reintroduce
  wall-based blocking or "see through windows" behaviour this milestone.
- Demo scale reference: the reference rules use ~6"×4" / 12"×6" ruins on a 44"×60" board; scale down to the
  project's board sizes for the example config.
</specifics>

<deferred>
## Deferred Ideas

- **Walls** (thin L-shaped wall segments within a footprint): revisit in a later milestone for
  rendering realism, movement interaction, and/or finer-grained LOS. Explicitly NOT in this
  milestone — the reference LOS model is footprint-based.
- **Cover / Benefit of Cover** ("not fully visible because of terrain" → +1 save): maps cleanly onto
  the footprint model later; deferred.
- **Dense/Woods visibility, difficult/impassable movement, elevation & Plunging Fire, board
  templates/procedural placement, variable base sizes** — later terrain milestone(s).
- **Terrain in observation** — Phase 2 of this milestone.

### Reviewed Todos (not folded)
None — no pending todos matched this phase.
</deferred>

---

*Phase: 01-terrain-los-blocking*
*Context gathered: 2026-06-20*
