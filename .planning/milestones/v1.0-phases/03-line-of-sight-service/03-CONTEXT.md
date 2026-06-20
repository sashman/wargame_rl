# Phase 3: Line of Sight Service - Context

**Gathered:** 2026-04-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Introduce a **single domain-level line-of-sight (LOS) service** on the discrete grid: **Bresenham** (integer grid) ray tracing with a pluggable **blocking** predicate. The service must be **deterministic**, **unit-tested** on boards with known blocking layouts, and **callable from** future rules, action masks (Phase 4), and renderers (LOS-04) without duplicating logic.

Phase 3 **does not** add shooting, weapon range, or action-mask wiring (LOS-03 is Phase 4). It **does not** implement full terrain v2; blocking is **terrain-style cells only**—**models do not occlude** LOS in this phase.

</domain>

<decisions>
## Implementation Decisions

### 1. Blocking data in v1 (before terrain milestone)

- **D-01:** Expose blocking through an **`is_blocking(x: int, y: int) -> bool`** (or equivalent `Callable[[int, int], bool]`) passed into LOS queries—not hard-coded global state. The core functions stay **pure** given `(board_width, board_height, is_blocking, ...)`.
- **D-02:** Add an **optional** way for the env to supply blocking from YAML/config: e.g. optional **`blocking_mask`** on `WargameEnvConfig` (or companion field) defaulting to **no blocking** (all cells clear). When present, the env builds a closure over the mask; **default configs keep current behaviour** (all clear).
- **D-03:** **Tests** use explicit small boards with hand-built masks or lambdas (fixtures)—no reliance on production YAML for every case.

### 2. Public API shape & DDD fit

- **D-04:** Implement the geometric core in **`wargame_rl/wargame/envs/domain/los.py`** (name may vary): **pure functions** with no Gym imports.
- **D-05:** Primary entry points:
  - **`has_line_of_sight(...)`** → `bool`
  - **`iter_los_cells(...)`** (or `cells_along_los`) → **iteration or sequence** of `(x, y)` cells along the same trace as the boolean check, for **debug/render** (LOS-04).
- **D-06:** Callers pass **`board_width`**, **`board_height`**, and **`is_blocking`**. **`WargameEnv` / `Battle`** wires dimensions from existing state and the mask from config; **`BattleView`** consumers that need LOS use the env or a small **facade** that closes over the current `is_blocking`—avoid duplicating mask logic outside domain/env wiring.
- **D-07:** Model-index-based LOS (`shooter_idx`, `target_idx`) is **not** required in Phase 3; Phase 4 can map indices to `(x, y)` and call the grid API.

### 3. Grid trace semantics (Bresenham)

- **D-08:** Use **2D integer Bresenham** between cell centres on the **square grid**; coordinates match existing env convention **`location[0] == x`**, **`location[1] == y`** (see renderer usage).
- **D-09:** **Same cell** `(x0,y0) == (x1,y1)` → **`True`** (trivial clear LOS for API consistency).
- **D-10:** **Blocking evaluation** applies only to **strictly interior** cells along the traced segment: **exclude** the **shooter cell** and **exclude** the **target cell** from `is_blocking` checks. Any interior cell with `is_blocking True` → **no LOS**.
- **D-11:** **Endpoints out of bounds** → **`False`** or documented raise: prefer **`False`** for “no LOS” after clamping/validation (planner picks one; stay consistent in tests).
- **D-12:** **Diagonals** use the same Bresenham trace (no special “corner cutting” rules beyond the chosen algorithm—document the reference implementation or cite a standard formulation in module docstring).

### 4. Model occlusion vs terrain-only blocking

- **D-13:** Phase 3 **does not** treat **intervening models** (player or opponent, alive or dead) as blockers. Only **`is_blocking`** from terrain/mask applies.
- **D-14:** A **later phase** may compose `is_blocking = terrain_mask OR model_occupancy` for tabletop fidelity; that is **explicitly deferred**—note in roadmap/deferred, not implemented here.

### 5. Renderers and reuse (LOS-04 prep)

- **D-15:** **Human / debug render** should use **`iter_los_cells`** (or equivalent) from the **same module** as `has_line_of_sight` so the drawn line matches the logical test.

### Claude's Discretion

- Exact field name and Pydantic shape for optional mask on `WargameEnvConfig` (1D sparse vs 2D dense vs run-length); ndarray layout **`(board_height, board_width)` vs `(width, height)`** as long as `is_blocking(x,y)` matches **`model.location`** indexing.
- Whether `iter_los_cells` returns `list[tuple[int,int]]` or a numpy array.
- Minor optimisations (early exit on first blocker) provided behaviour matches spec.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements and roadmap

- `.planning/REQUIREMENTS.md` — LOS-01, LOS-02, LOS-04
- `.planning/ROADMAP.md` — Phase 3 goal and success criteria (Bresenham, single service, deterministic tests with blocking cells)

### Prior phase context (dependency direction, grid, slots)

- `.planning/phases/01-wounds-elimination/01-CONTEXT.md` — domain vs env, `BattleView`, fixed model indices
- `.planning/phases/02-alive-aware-observation/02-CONTEXT.md` — observation pipeline; no LOS in obs this phase

### Architecture

- `docs/ddd-envs.md` — domain layer, `BattleView`, no Gym inside pure domain modules

### Code touchpoints (expected)

- `wargame_rl/wargame/envs/domain/battle.py` — dimensions; optional future holder for mask reference if needed
- `wargame_rl/wargame/envs/domain/battle_view.py` — protocol; extend only if necessary for render—prefer wiring via env/facade
- `wargame_rl/wargame/envs/types/config.py` — optional blocking config defaulting to none
- `wargame_rl/wargame/envs/wargame.py` — construct `is_blocking` for LOS helpers
- `wargame_rl/wargame/envs/renders/human.py` — optional LOS overlay using `iter_los_cells`

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable assets

- `BoardDimensions` / `WargameEnvConfig.board_width` & `board_height` — authoritative size for bounds checks.
- `WargameModel.location` as `(x, y)` integer grid coords — aligns LOS endpoints with movement and rendering.

### Established patterns

- New **domain** behaviour lives under `domain/` with **no** `gymnasium` imports in pure rule modules (`docs/ddd-envs.md`).
- Config fields default to **no-op** so existing YAML unchanged (`PROJECT.md` constraints).

### Integration points

- Phase 4 action masks will call LOS with shooter/target positions and weapon range (separate concern).
- Phase 5+ shooting resolution consumes “can see target” from the same service.

</code_context>

<specifics>
## Specific Ideas

User selected discussion areas **1–4** (blocking in v1, API shape, grid semantics, occlusion). Decisions above lock the recommended direction: **injectable blocking**, **pure `domain/los`**, **interior-cell-only blocking checks**, **no model occlusion in Phase 3**, shared **cell iterator** for render.

</specifics>

<deferred>
## Deferred Ideas

- **Model silhouettes / occupancy blocking LOS** — future phase; compose `is_blocking` then.
- **Dense terrain v2** (cover, difficult ground, blocking semantics) — maps onto `is_blocking` implementation, not Phase 3 scope.
- **LOS-03** (action masking) — Phase 4.
- **Indirect fire / no-LOS shooting** — `.planning/REQUIREMENTS.md` Out of Scope table.

### Reviewed Todos (not folded)

- None — `todo match-phase 3` returned no pending matches.

</deferred>

---

*Phase: 03-line-of-sight-service*
*Context gathered: 2026-04-04*
