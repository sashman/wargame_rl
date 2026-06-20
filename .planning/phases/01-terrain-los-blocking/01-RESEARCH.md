# Phase 1: Terrain in the Simulation (LOS-Blocking) - Research

**Researched:** 2026-06-20
**Domain:** Brownfield discrete-grid RL wargame — footprint-based (10e Ruins) line-of-sight blocking through the existing DDD env + single Bresenham LOS seam
**Confidence:** **HIGH** (grounded in a full read of `domain/los.py`, `wargame.py`, `domain/battle.py`/`battle_factory.py`/`battle_view.py`/`value_objects.py`, `types/config.py`, `env_components/shooting_masks.py` + `observation_builder.py`, `renders/human.py`, `state/snapshot.py`, `tests/test_los.py`, and the milestone CONTEXT/REQUIREMENTS/ROADMAP)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions (from `## Implementation Decisions`)

**LOS model — canonical 10e Ruins (footprint-based):**
- **D-01:** The **footprint** is the LOS primitive. A ruin blocks LOS between observer O and target T **iff** the line O→T passes through that ruin's footprint **AND both O and T are outside that footprint**.
- **D-02:** **See-into exception** — if T is inside the footprint, T is visible to O.
- **D-03:** **See-out exception** — if O is inside the footprint, O can see out normally. (Two models inside the same footprint always see each other.)
- **D-04:** The exception is **per-ruin** — being inside ruin A's footprint does **not** let you see through a different ruin B. Each ruin is evaluated independently.
- **D-05:** **Walls play NO role in LOS this milestone.** Do not reintroduce wall-based blocking or "see through windows" behaviour. Walls are deferred.
- **D-06:** Movement is unaffected — models may move through and occupy footprint cells freely.
- **D-07:** Blocking is **endpoint-aware** — the injectable `is_blocking(x, y)` predicate must be built **per query** from the endpoints (a cell is blocking iff it is inside some footprint F with O∉F and T∉F). The `domain/los.py` Bresenham core stays unchanged. This generalises (does not replace) the existing `blocking_mask`, which keeps working as an always-on blocking grid.

**Config schema:**
- **D-08:** A footprint is two opposite corners `[x0, y0, x1, y1]` in **absolute** board coordinates. No wall config this milestone.
- **D-09:** Terrain config defaults to **None/empty** (no-op). Existing YAML and pre-terrain checkpoints keep working unchanged.

**Placement & validation:**
- **D-10:** Footprint corners must be validated **within board bounds** at load (clear error otherwise).
- **D-11:** Footprints **may overlap** deployment zones and objectives, and models **may spawn on / occupy** footprint cells.
- **D-12:** Footprints must **not overlap each other** — validated at load.

**Rendering:**
- **D-13:** Draw each footprint as a **translucent filled rectangle with an outline + label**.
- **D-14:** Colour the debug LOS line by the actual **blocked/clear verdict**.

**Demo scenario:**
- **D-15:** Ship a **new example env config**: two facing deployment zones with one or two ruins between them that block cross-board sightlines.

### Claude's Discretion
- Exact footprint rasterisation / "passes through footprint" cell test (which interior cells count), endpoint canonicalisation for LOS symmetry, and the precise rendering colours/alpha — planner/implementer decides, constrained by the required tests.
- Whether footprints are stored as rectangles + a derived membership test vs a precomputed boolean footprint grid — implementer's call (performance/clarity).

### Deferred Ideas (OUT OF SCOPE)
- **Walls** (thin L-shaped segments within a footprint) — explicitly NOT this milestone.
- **Cover / Benefit of Cover** (+1 save).
- **Dense/Woods visibility, difficult/impassable movement, elevation & Plunging Fire, board templates/procedural placement, variable base sizes.**
- **Terrain in observation** — Phase 2 (do NOT plan any observation/tensor/network work here).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TERR-01 | Ruin configurable in YAML as footprint rect `[x0,y0,x1,y1]` (absolute coords) | New Pydantic `TerrainPieceConfig` + `WargameEnvConfig.terrain` field, mirroring `deployment_zone` tuple + `ObjectiveConfig` patterns (§Standard Stack, §Code Examples) |
| TERR-02 | Validated at load — corners in bounds, footprints non-overlapping | `model_validator(mode="after")` mirroring `validate_blocking_mask_shape` / `_validate_entity_configs` (§Code Examples, Pitfall: validation) |
| TERR-03 | No-terrain configs behave exactly as today (no-op default) | `terrain=None` default; `blocking_mask`/`number_of_opponent_models=0` precedent; byte-identical golden traces (§Common Pitfalls #3, §Validation Architecture) |
| TERR-04 | Footprint blocks LOS when between two outside models; see-out / see-into per ruin | Endpoint-aware predicate built in `has_line_of_sight_between_cells`; `domain/terrain.py` membership helper (§Architecture Patterns, §Code Examples) |
| TERR-05 | Terrain does not affect movement | Grid/footprints consumed **only** by the LOS predicate; nothing in `ActionHandler.apply`/placement reads it (§Architecture Patterns, §Don't Hand-Roll) |
| TERR-06 | Footprint LOS flows through the single LOS service so masks/resolution/render agree | `_make_is_blocking` endpoint-aware; all consumers already route through `view.has_line_of_sight_between_cells` (verified, §Architecture Patterns, Pitfall #4) |
| TERR-07 | LOS symmetric and deterministic on known boards | Endpoint canonicalisation in the seam + Hypothesis symmetry property test (§Common Pitfalls #2, §Validation Architecture) |
| TERR-10 | Renderer draws footprints (translucent fill + outline + label); debug LOS coloured by verdict | `renders/human.py` overlay from `BattleView.terrain` + verdict-coloured `_draw_debug_los_line` (§Architecture Patterns, Pitfall #5) |

*(TERR-08/TERR-09 = observation; Phase 2 — explicitly out of scope here.)*
</phase_requirements>

## Project Constraints (from .cursor/rules/)

These are authoritative and rank with the locked decisions above. The planner must not propose anything that violates them.

- **`python-conventions.mdc`:** all functions typed (mypy **strict**), `from __future__ import annotations`, modern generics (`list[int]`, `str | None`), isort Black-profile absolute imports (`from wargame_rl.wargame...`), Ruff 88 cols / 4-space / double quotes, Pydantic for config, `numpy` typed arrays for perf.
- **`gymnasium-env.mdc`:** typed Pydantic obs/action/info/config; "Adding Features" checklist (config → env_components/logic → obs → tensor → networks → reward/renderer → tests + backward compat — **Phase 1 stops at logic + renderer + tests**; obs/tensor/networks are Phase 2); new entities **mirror existing patterns**, always backward compatible; new config fields default to a **no-op**; renderer uses `BattleView` only.
- **`dev-workflow.mdc` / CLAUDE.md:** `uv` only (`uv add` for deps, never edit `uv.lock`); `just validate` (format + lint + test) before pushing; run `uv run ruff format` + `uv run ruff check --fix` + `uv run pre-commit run --files <changed>` before declaring done; feature branch (`feature/<topic>`), never commit to `main`; public methods get docstrings (WHAT); comments explain WHY.
- **`docs/ddd-envs.md`:** dependency direction is **domain → types only**; domain must NOT import `env_components`, `reward`, or `renders`. `Battle` is the aggregate; `BattleView` is the read-only projection; reward/renders depend on `BattleView`, not the env. New entity recipe: define in `domain/` → add config in `types/` → wire `battle_factory` → expose on `Battle` + `BattleView` → (obs later) → backward-compat re-exports.
- **`testing.mdc`:** Pytest + `conftest.py` fixtures; prefer integration tests through public APIs (add properties rather than touching `_private`); type-annotate tests; cover config validation · env integration · backward compat; `just validate` before push.

## Summary

This phase adds **Ruins** that block line of sight using the canonical Warhammer 40k 10e **footprint** abstraction: a ruin is a rectangle `[x0,y0,x1,y1]` and the footprint *itself* is the LOS blocker, with **see-into** (target inside ⇒ visible) and **see-out** (observer inside ⇒ sees) exceptions evaluated **per ruin**. The codebase is exceptionally well-prepared for this: it already has a single Bresenham LOS service (`domain/los.py`) with an **injectable `is_blocking(x, y)` predicate**, and a single seam (`WargameEnv.has_line_of_sight_between_cells`) through which **every** consumer (shooting masks, action masks, shooting resolution, renderer debug line) already routes. The one behavioural change is to make the predicate **endpoint-aware**: filter footprints to those containing neither endpoint, then a cell blocks iff the existing `blocking_mask` blocks it OR it lies inside one of those filtered footprints. `domain/los.py` (Bresenham + interior-only loop) does **not** change.

The work follows the existing DDD spine identically to objectives: new Pydantic `TerrainPieceConfig` + `WargameEnvConfig.terrain` (default `None`) → a new **pure** `domain/terrain.py` (footprint value objects + membership/blocking helpers, importing `types/` only) → wire `battle_factory.from_config` → expose `Battle.terrain` + `BattleView.terrain` → repoint `WargameEnv._make_is_blocking` to be endpoint-aware → renderer overlay. The single highest-value test is the **Hypothesis LOS-symmetry property test** (`los(A,B) == los(B,A)`), because Bresenham is direction-dependent near cell corners; the cheap, robust mitigation is to **canonicalise endpoint order** inside the seam before tracing. Backward compatibility is structural: `terrain=None` ⇒ empty footprint list ⇒ the predicate degenerates to today's `blocking_mask`-only closure ⇒ `tests/test_los.py` golden traces stay **byte-identical**.

There are **no new dependencies** (pure Python + NumPy + existing Pydantic) and **no observation/tensor/network work** (that is Phase 2). The main risks are all geometry/contract-integrity issues, each with a concrete test: LOS asymmetry, mask↔resolution divergence, interior-only contract violation, and renderer/logic divergence.

**Primary recommendation:** Add `domain/terrain.py` (`Footprint` frozen dataclass + `Terrain` collection with `contains`/`blocking_footprints_for_endpoints`), store it on `Battle`, expose via `BattleView.terrain`, and rewrite `_make_is_blocking` to take the four endpoint coords and return a per-query closure `blocking_mask OR (cell ∈ any footprint not containing either endpoint)`. Canonicalise `(x0,y0,x1,y1)` order in `has_line_of_sight_between_cells` to guarantee symmetry. Keep `domain/los.py` untouched.

## Standard Stack

No new packages. Everything needed is already in the project.

### Core
| Library | Version (locked in `uv.lock`) | Purpose | Why standard here |
|---------|------|---------|--------------|
| Python | 3.13 (project floor 3.12+) | Frozen dataclasses, `list[tuple[int,int]]` generics | Project standard |
| pydantic | 2.x (already used) | `TerrainPieceConfig`, `WargameEnvConfig.terrain` field + validators | All env config is Pydantic; mirror `ObjectiveConfig` / `blocking_mask` |
| numpy | already used | Optional precomputed boolean footprint grid; deployment-zone-style arrays | Project uses typed numpy arrays for perf |
| pygame | already used | Footprint overlay (translucent rect + outline + label), verdict-coloured LOS line | Existing `renders/human.py` renderer |

### Supporting
| Component | Location | Purpose | When to use |
|---------|----------|---------|-------------|
| `domain/los.py::has_line_of_sight` | existing, **unchanged** | Bresenham + injectable `is_blocking`, interior-only | The footprint predicate plugs straight in |
| `pytest` + `conftest.py` fixtures | existing | golden boards, validation, mask consistency | All Phase-1 tests |
| `hypothesis` | **verify availability — see §Environment Availability** | LOS-symmetry property test (`los(A,B)==los(B,A)`) | The single highest-value test |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Per-query footprint membership (rect `contains`) | Precomputed per-footprint boolean grids | Footprints are few (1–2) and static; rect `contains` is O(1) and clearer. A precomputed grid only helps if many footprints; **implementer's call (D-discretion)**. The base `blocking_mask` can still be read directly (already static). |
| Canonicalise endpoints in the seam | Make `los.py` Bresenham symmetric | Canonicalisation is a 1-line change in `has_line_of_sight_between_cells`, keeps `los.py` (and its golden traces) untouched. Strongly preferred. |
| `terrain: list[TerrainPieceConfig]` | reuse `blocking_mask` for footprints | Footprints need the endpoint-aware see-in/see-out rule, which a static mask cannot express. Must be a distinct, structured field. |

**Installation:** none. If `hypothesis` is not already a dev dependency, `uv add --dev hypothesis` (never edit `uv.lock`).

## Architecture Patterns

### Recommended file layout (mirrors `objectives`)
```
wargame_rl/wargame/envs/
├── domain/
│   ├── terrain.py        # NEW: Footprint (frozen) + Terrain collection + membership/blocking helpers (pure; imports types/ only)
│   ├── battle.py         # MOD: hold `terrain`; expose `Battle.terrain`
│   ├── battle_factory.py # MOD: build Terrain from config.terrain; pass to Battle
│   └── battle_view.py    # MOD: add `terrain` property to the Protocol
├── types/config.py       # MOD: TerrainPieceConfig + WargameEnvConfig.terrain (default None) + validators
├── wargame.py            # MOD: _make_is_blocking endpoint-aware; `terrain` property; canonicalise in seam
└── renders/human.py      # MOD: footprint overlay + verdict-coloured debug LOS line
examples/env_config/
└── <new>_terrain_los.yaml # NEW (D-15)
```

### Pattern 1: Endpoint-aware blocking predicate (the one behavioural edit)
**What:** The blocking predicate must depend on which footprints the two query endpoints sit in. Build it **per query** from the endpoints; keep `domain/los.py` untouched.
**When to use:** Inside `WargameEnv.has_line_of_sight_between_cells` (and therefore for every consumer that routes through it).
**Pattern:**
```python
# wargame.py — endpoint-aware seam (illustrative; types per mypy strict)
def _make_is_blocking(
    self, x0: int, y0: int, x1: int, y1: int
) -> Callable[[int, int], bool]:
    """Per-query blocking predicate: static blocking_mask OR footprint membership,
    where footprints containing either endpoint are skipped (10e see-out / see-into)."""
    mask = self.config.blocking_mask
    active = self._battle.terrain.blocking_footprints_for_endpoints(x0, y0, x1, y1)

    def is_blocking(x: int, y: int) -> bool:
        if mask is not None and mask[y][x]:
            return True
        return any(fp.contains(x, y) for fp in active)

    return is_blocking

def has_line_of_sight_between_cells(self, x0, y0, x1, y1) -> bool:
    # Canonicalise endpoint order so los(A,B) traces the same cells as los(B,A)
    (ax, ay), (bx, by) = sorted([(x0, y0), (x1, y1)])
    return has_line_of_sight(
        ax, ay, bx, by, self.board_width, self.board_height,
        self._make_is_blocking(ax, ay, bx, by),  # predicate is endpoint-symmetric anyway
    )
```
> Note: footprint membership (`O∉F and T∉F`) is **already symmetric** in O,T. The residual asymmetry is purely the Bresenham *trace*, fixed by canonicalising endpoint order. Apply canonicalisation in the seam so masks, resolution, and renderer all inherit it.

### Pattern 2: Pure domain `terrain.py` (DDD — imports `types/` only)
**What:** `Footprint` frozen dataclass with a normalised (min/max) rectangle and an O(1) `contains`; a `Terrain` collection with the endpoint filter. No Gym, no Bresenham, no `env_components`/`reward`/`renders` imports.
```python
# domain/terrain.py (illustrative)
@dataclass(frozen=True, slots=True)
class Footprint:
    x0: int; y0: int; x1: int; y1: int  # stored normalised: x0<=x1, y0<=y1
    def contains(self, x: int, y: int) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

class Terrain:
    def __init__(self, footprints: list[Footprint]) -> None: ...
    @property
    def footprints(self) -> list[Footprint]: ...
    def blocking_footprints_for_endpoints(
        self, x0: int, y0: int, x1: int, y1: int
    ) -> list[Footprint]:
        """Footprints that can block this query: neither endpoint inside (D-01..D-04)."""
        return [
            fp for fp in self._footprints
            if not fp.contains(x0, y0) and not fp.contains(x1, y1)
        ]
```
**Decision needed (D-discretion):** footprint `contains` is **inclusive of both corners** (footprint cells are every integer cell in the rect). Confirm and pin in tests. Because `los.py` checks only **strictly-interior** segment cells, a footprint edge cell that happens to be an endpoint is excluded automatically — no special-casing.

### Pattern 3: DDD wiring (mirror objectives exactly)
- `types/config.py`: `TerrainPieceConfig(BaseModel)` with `footprint: tuple[int,int,int,int]` (or `x0/y0/x1/y1`); `WargameEnvConfig.terrain: list[TerrainPieceConfig] | None = None`.
- `battle_factory.from_config`: build `list[Footprint]` (normalise corners) → `Terrain(...)` → pass into `Battle`.
- `battle.py`: store `self._terrain`; add `terrain` property. `reset_for_episode` leaves terrain intact (static per episode).
- `battle_view.py`: add `terrain` to the `BattleView` Protocol; `WargameEnv.terrain` delegates to `self._battle.terrain`.

### Pattern 4: Renderer overlay from `BattleView.terrain` (single source of truth)
- Draw each footprint as a translucent filled rect + outline + label (D-13) **from `view.terrain`** (the same data LOS uses), not from a parallel source.
- Colour `_draw_debug_los_line` by `view.has_line_of_sight_between_cells(...)` verdict (D-14): e.g. green = clear, red = blocked; optionally mark the first interior blocking cell. The renderer already only *traces* via `iter_los_cells`; add a verdict call to colour it.

### Anti-Patterns to Avoid
- **Editing `domain/los.py`** to "support footprints" → breaks the interior-only contract and golden traces. Keep it untouched; all change is in the predicate.
- **Reintroducing walls / "see through windows"** → explicitly forbidden (D-05).
- **A consumer building its own blocking predicate** (resolution, renderer, snapshot) → divergence. Everything must route through `has_line_of_sight_between_cells`.
- **Making `terrain` required / non-defaulted** → breaks every existing YAML. Default `None`.
- **A fully static precomputed blocking grid** for footprints → wrong: blocking is endpoint-aware, so the footprint contribution **cannot** be baked into a single global grid (only the `blocking_mask` part is static).
- **Footprint affecting movement / placement** → LOS-only this milestone; nothing but the LOS predicate may read terrain.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Ray traversal across the grid | A new line/supercover algorithm | `domain/los.has_line_of_sight` (unchanged) | One Bresenham path already exists, is tested, and is the single seam |
| "Is this cell on the line" | Re-tracing in the predicate/renderer | The existing interior-only loop in `los.py` consumes the predicate | Endpoint-aware logic belongs in `is_blocking`, not a second trace |
| LOS fan-out to consumers | Calling LOS in masks/resolution/render separately | `view.has_line_of_sight_between_cells` (all already route here) | Zero call-site changes; guarantees agreement (TERR-06) |
| Rectangle-in-bounds / overlap checks | Ad-hoc inline checks scattered around | A `model_validator(mode="after")` mirroring `validate_blocking_mask_shape` | Validation-at-construction is the project convention; one clear error site |
| Property-based symmetry testing | Hand-written nested loops over a few boards | `hypothesis` strategies over random boards/endpoints/footprints | Catches corner-grazing asymmetry the few golden boards miss |

**Key insight:** the entire LOS subsystem is already a single, injectable, cell-granular seam. The whole phase is "compute the right `is_blocking` closure and hand it to the existing machinery," plus DDD wiring and a renderer overlay. Resist adding any new geometry algorithm.

## Common Pitfalls

### Pitfall 1: LOS symmetry breaks (`A→B` clear but `B→A` blocked)
**What goes wrong:** Shooting masks query LOS shooter→target; if `los(A,B) ≠ los(B,A)`, the agent gets one-way firing lanes and RL will exploit them.
**Why it happens:** `_bresenham_line(A,B)` and `(B,A)` can traverse *different* interior cells near a cell corner. Footprint membership is symmetric, but the **trace** is not, so a footprint grazing one trace and not the other flips the verdict.
**How to avoid:** (1) **Hypothesis property test** `has_line_of_sight(A,B)==has_line_of_sight(B,A)` over random boards/endpoints/footprints (highest-value test). (2) **Canonicalise endpoint order** in `has_line_of_sight_between_cells` (sort the two points) so both directions trace the identical cell set. Keep it in the seam so all consumers inherit it.
**Warning signs:** agent shoots from "one-way" cells near footprint corners; symmetry test fails on a seed.

### Pitfall 2: Interior-only / endpoint contract violated
**What goes wrong:** A model standing **inside or on** a footprint loses the ability to shoot, or a "fix" makes endpoints inclusive and breaks every golden trace.
**Why it happens:** Movement passes through footprints (D-06), so a model can stand inside one. The 10e rule already says see-out/see-into (D-02/D-03), which the endpoint filter handles; and `los.py` checks only `cells[1:-1]` (endpoints excluded), so a footprint cell that is an endpoint is ignored by construction.
**How to avoid:** **Do not touch** `los.py`'s interior-only loop. Add a regression test: a footprint on the shooter's or target's exact cell does **not** block (extend `test_los_interior_only_blocking_ignores_endpoint_blocker` with a terrain-derived predicate). Keep golden traces byte-identical when no terrain is configured.
**Warning signs:** existing `test_los.py` tests fail after the change; models inside a ruin can't shoot out.

### Pitfall 3: No-terrain backward-compat regression
**What goes wrong:** Adding the field or rewriting `_make_is_blocking` subtly changes behaviour even when `terrain=None`.
**Why it happens:** The predicate path changed shape; if the empty-terrain branch isn't a clean degenerate of today, traces drift.
**How to avoid:** `terrain=None` ⇒ `Terrain([])` ⇒ `blocking_footprints_for_endpoints` returns `[]` ⇒ predicate reduces to exactly `mask[y][x]` (today). Add a test asserting `test_los.py` golden traces and `test_wargame_env_los_uses_config_mask` stay green; assert a no-terrain config's LOS matrix is identical to a pre-change baseline.
**Warning signs:** `test_los.py` / `test_integration` fail; a no-terrain LOS query flips.

### Pitfall 4: Mask vs resolution LOS divergence
**What goes wrong:** The shooting mask (what the agent may select) and resolution (what happens) disagree → train/serve skew.
**Why it happens:** A consumer builds its own predicate instead of the seam. (Today resolution does **not** re-check LOS in `_resolve_shooting_action` — it trusts the mask — so the main risk is the *mask* path and any future re-check diverging.)
**How to avoid:** Keep `has_line_of_sight_between_cells` the **only** LOS entry point; `compute_shooting_masks` already takes `view.has_line_of_sight_between_cells`. Add a consistency test: for a board with a footprint, every mask-permitted `(shooter,target)` pair is LOS-clear ∩ in-range ∩ alive, and vice-versa. Grep for direct `has_line_of_sight(` / `blocking_mask` use outside the seam — there should be none in consumers.
**Warning signs:** a selected shoot action no-ops; consistency test fails with terrain.

### Pitfall 5: Renderer / debug LOS diverges from domain truth
**What goes wrong:** The overlay shows a footprint in one place but shots block elsewhere, or the debug line reads "clear" while resolution blocks.
**Why it happens:** `_draw_debug_los_line` currently traces via `iter_los_cells` and **does not** apply blocking; if the footprint overlay is drawn from a different source than `view.terrain`, visual ≠ logical.
**How to avoid:** Draw the overlay from `view.terrain` (same data as LOS). Colour the debug line by the actual `view.has_line_of_sight_between_cells(...)` verdict (D-14). Optionally mark the first interior blocking cell.
**Warning signs:** a blocked shot drawn as a clear (green/uncoloured) line; footprint rectangle not aligned with where shots block.

### Pitfall 6 (awareness only — out of strict scope): wall-camping / distribution shift
**What goes wrong:** Once footprints block LOS, the policy may find one-way safe cells or camp behind a ruin and ignore objectives; enabling terrain on a no-terrain checkpoint conflates a distribution shift with bugs.
**How to avoid (flag for curriculum/eval, not this phase):** ship terrain as a **distinct example config** (D-15), keep a sparse objective anchor in any terrain curriculum phase, and confirm the no-terrain path is byte-identical so regressions are attributable. Do **not** add a terrain/cover reward here.

## Code Examples

Verified against the current codebase (patterns to mirror).

### Config field + validator (mirror `blocking_mask` / `ObjectiveConfig` / `_validate_entity_configs`)
```python
# types/config.py (illustrative; full typing per mypy strict)
class TerrainPieceConfig(BaseModel):
    """A ruin footprint as two opposite corners [x0, y0, x1, y1] in absolute coords."""
    footprint: tuple[int, int, int, int] = Field(
        description="Footprint rectangle (x0, y0, x1, y1) in absolute board cells."
    )

# on WargameEnvConfig:
terrain: list[TerrainPieceConfig] | None = Field(
    default=None,
    description="Optional ruin footprints that block LOS. None = no terrain (no-op).",
)

@model_validator(mode="after")
def validate_terrain(self) -> "WargameEnvConfig":
    if self.terrain is None:
        return self
    rects = [self._normalise(t.footprint) for t in self.terrain]
    for i, (x0, y0, x1, y1) in enumerate(rects):
        if x0 < 0 or y0 < 0 or x1 >= self.board_width or y1 >= self.board_height:
            raise ValueError(
                f"terrain[{i}] {self.terrain[i].footprint} is outside the board "
                f"({self.board_width}x{self.board_height})"
            )
    for i in range(len(rects)):           # D-12: footprints must not overlap each other
        for j in range(i + 1, len(rects)):
            if _rects_overlap(rects[i], rects[j]):
                raise ValueError(f"terrain[{i}] overlaps terrain[{j}]")
    return self
```
*(Reference patterns: `WargameEnvConfig.validate_blocking_mask_shape` lines ~414–430, `_validate_entity_configs` lines ~30–56, `deployment_zone: tuple[int,int,int,int]` line ~222.)*

### Example demo config (D-15)
```yaml
# examples/env_config/<name>_terrain_los.yaml — two facing zones, ruins between them
config_name: terrain_los_demo
board_width: 60
board_height: 44
number_of_wargame_models: 4
number_of_opponent_models: 4
number_of_objectives: 3
deployment_zone: [0, 0, 20, 44]
opponent_deployment_zone: [40, 0, 60, 44]
opponent_policy:
  type: scripted_advance_to_objective
terrain:
  - { footprint: [27, 8, 33, 16] }    # ~6x8 ruin blocking the north sightline
  - { footprint: [27, 28, 33, 36] }   # ~6x8 ruin blocking the south sightline
objectives:
  - { x: 30, y: 6 }
  - { x: 30, y: 22 }
  - { x: 30, y: 38 }
```
*(Scale reference, CONTEXT canonical refs: 40k uses ~6"×4" / 12"×6" ruins on a 44"×60" board; scale to the project's grid.)*

## Runtime State Inventory

Not applicable — this is a greenfield **feature addition**, not a rename/refactor/migration. No stored data, live-service config, OS-registered state, secrets, or build artifacts encode any renamed string. **None — verified: no string rename or data migration is involved.**

## Environment Availability

| Dependency | Required by | Available | Notes |
|------------|------------|-----------|-------|
| Python 3.12+/3.13, numpy, pydantic, pygame | core implementation | ✓ | All already in `pyproject.toml` / `uv.lock` |
| `pytest` | all tests | ✓ | Existing suite |
| `hypothesis` | LOS-symmetry property test (TERR-07) | **VERIFY** | Plan a Wave-0 check: `uv run python -c "import hypothesis"`. If missing, `uv add --dev hypothesis` (never edit `uv.lock`). |

**Missing with no fallback:** none — even if `hypothesis` is absent, the property test can fall back to a deterministic seeded random loop, but `hypothesis` is strongly preferred (project already uses property/parameterised testing per CLAUDE.md).

## Validation Architecture

Nyquist validation is **enabled** (`.planning/config.json → workflow.nyquist_validation: true`).

### Test Framework
| Property | Value |
|----------|-------|
| Framework | `pytest` (+ `hypothesis` for the property test) |
| Config file | `pyproject.toml` (pytest config); fixtures in `tests/conftest.py` |
| Quick run command | `uv run pytest tests/test_los.py -x` |
| Full suite command | `just test` (or `uv run pytest`) |
| Full validation | `just validate` (format + lint + test) before push |

### Test surfaces (what to add / extend)
1. **`tests/test_terrain.py` (NEW) — domain `Footprint`/`Terrain` units (no Gym):**
   - `Footprint.contains` inclusive of both corners; normalisation of unordered corners.
   - `blocking_footprints_for_endpoints` excludes footprints containing either endpoint (D-01..D-04).
2. **`tests/test_los.py` (EXTEND) — golden boards through the env seam:**
   - **Blocked:** two models outside a footprint that lies between them ⇒ `has_line_of_sight_between_cells` is `False`.
   - **See-into:** target inside the footprint ⇒ `True`.
   - **See-out:** observer inside the footprint ⇒ `True`.
   - **Per-ruin:** observer inside ruin A but ruin B lies between O and T ⇒ `False` (A's exception doesn't apply to B).
   - **Off-line footprint:** footprint not on the segment ⇒ unaffected (`True`).
   - **Interior-only regression:** footprint cell coinciding with an endpoint does not block (extend `test_los_interior_only_blocking_ignores_endpoint_blocker`).
   - **Backward-compat:** existing golden traces (`test_los_golden_trace_zero_three_one`, `test_wargame_env_los_uses_config_mask`) stay **byte-identical** with no terrain.
   - **blocking_mask + footprint co-existence:** a config with both still blocks via OR.
3. **LOS-symmetry property test (NEW, `hypothesis`):** over random board sizes, footprints, and endpoint pairs, assert `has_line_of_sight_between_cells(A,B) == (B,A)`. This is the **single highest-value test** (TERR-07).
4. **`tests/test_*` config validation (NEW/extend, e.g. in `test_los.py` or `test_fixed_placement.py`):**
   - Off-board footprint corner ⇒ `ValueError` (TERR-02/D-10).
   - Two overlapping footprints ⇒ `ValueError` (TERR-02/D-12).
   - Footprint overlapping a deployment zone / objective is **allowed** (D-11).
   - `terrain=None` default on an existing fixture config (TERR-03).
5. **Mask↔resolution consistency (extend `test_shooting_resolution.py` / `test_env`):** with a footprint between shooter and target, the shooting mask forbids the shot; with see-into/see-out it permits; mask-permitted pairs == LOS-clear ∩ range ∩ alive (TERR-06, Pitfall #4).
6. **Movement-unaffected (extend `test_env`):** a model can move into / through / occupy a footprint cell (TERR-05). Confirm placement and `ActionHandler.apply` ignore terrain.
7. **Renderer smoke (optional, light):** rendering a terrain config does not raise; debug-LOS verdict colour reflects `has_line_of_sight_between_cells` (TERR-10). Keep light — pygame surface assertions are brittle; assert the overlay reads from `view.terrain` and the verdict helper returns the same bool as the seam.
8. **Snapshot note:** `state/snapshot.py` does **not** currently compute LOS (verified — no `has_line_of_sight` call), so no snapshot change is required for blocking to "agree"; it agrees vacuously. No snapshot test needed this phase unless a snapshot LOS field is added (it is not).

### Phase Requirements → Test Map
| Req | Behavior | Test type | Automated command | Exists? |
|-----|----------|-----------|-------------------|---------|
| TERR-01 | YAML footprint parses to config | unit | `uv run pytest tests/test_los.py -k terrain_config -x` | ❌ Wave 0 |
| TERR-02 | Out-of-bounds / overlap rejected | unit | `uv run pytest tests/test_los.py -k terrain_validation -x` | ❌ Wave 0 |
| TERR-03 | No-terrain byte-identical | regression | `uv run pytest tests/test_los.py -x` | ⚠️ extend existing golden tests |
| TERR-04 | block / see-in / see-out / per-ruin / off-line | integration | `uv run pytest tests/test_los.py -k terrain_los -x` | ❌ Wave 0 |
| TERR-05 | movement through footprint | integration | `uv run pytest tests/test_env.py -k terrain_movement -x` | ❌ Wave 0 |
| TERR-06 | mask == resolution with terrain | integration | `uv run pytest tests/test_shooting_resolution.py -k terrain -x` | ❌ Wave 0 |
| TERR-07 | LOS symmetric (random) | property | `uv run pytest tests/test_los.py -k symmetry -x` | ❌ Wave 0 |
| TERR-10 | renderer overlay + verdict colour | smoke | `uv run pytest tests/test_render*.py -k terrain -x` (or light unit on verdict helper) | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/test_los.py tests/test_terrain.py -x` (fast LOS + domain units).
- **Per wave merge:** `just test` (full suite — backward compat, env, shooting, render).
- **Phase gate:** `just validate` green (format + lint + full test) before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `tests/test_terrain.py` — `Footprint`/`Terrain` domain units (covers TERR-04 building blocks).
- [ ] Footprint/terrain golden boards added to `tests/test_los.py` (TERR-04, TERR-03 regression, interior-only).
- [ ] LOS-symmetry `hypothesis` property test (TERR-07) — confirm/install `hypothesis`.
- [ ] Config-validation tests (TERR-01/TERR-02) — bounds + non-overlap + allowed-overlap-with-zone/objective.
- [ ] Mask↔resolution consistency test with terrain (TERR-06).
- [ ] Movement-through-footprint test (TERR-05).
- [ ] (Light) renderer verdict-colour / overlay-source check (TERR-10).
- [ ] Framework check: `uv run python -c "import hypothesis"`; if absent `uv add --dev hypothesis`.

## State of the Art

| Old (earlier milestone draft) | Current (locked) | When changed | Impact |
|-------------------------------|------------------|--------------|--------|
| Walls rasterised whole-cell into a static `is_blocking` grid (`ARCHITECTURE.md`/`SUMMARY.md`) | **Footprint-based 10e Ruins LOS**, endpoint-aware predicate; walls deferred | 2026-06-20 (CONTEXT.md) | No wall rasterisation, no static footprint grid, no 4-connectivity/corner-pinhole concern; the predicate is per-query and endpoint-aware |

**Deprecated/superseded for this phase:**
- `ARCHITECTURE.md` §3–4 wall whole-cell rasterisation and `SUMMARY.md`'s wall recommendation — **superseded**. Reuse only their seam/backward-compat/test guidance.
- `PITFALLS.md` Pitfall 1 (L-wall corner pinhole) and the L-wall 4-connectivity blocker in `STATE.md` — **no longer apply** (no wall rasterisation). Pitfalls 2 (symmetry), 3 (interior-only), 4 (mask↔resolution), 8 (renderer divergence) **still apply** and are carried above.

## Open Questions

1. **Endpoint canonicalisation vs naturally symmetric footprints.**
   - What we know: footprint membership is symmetric; only the Bresenham trace is direction-dependent.
   - What's unclear: whether any footprint config actually produces asymmetry on the project's boards (it may be rare).
   - Recommendation: **canonicalise endpoint order in the seam unconditionally** (cheap, total) AND keep the property test. Don't rely on "probably symmetric."
2. **Footprint membership inclusivity at corners.**
   - What we know: `los.py` excludes endpoints; footprint `contains` should be inclusive of all rect cells.
   - Recommendation: inclusive both corners; pin with a `contains` unit test and a "footprint edge on the line blocks" golden board.
3. **Storage: rectangles vs precomputed boolean grid (D-discretion).**
   - Recommendation: start with rectangles + O(1) `contains` (few footprints, clearest, no memory blow-up); revisit only if profiling shows LOS hot (unlikely — 1–2 footprints).
4. **`hypothesis` availability.** Verify in Wave 0; `uv add --dev hypothesis` if missing.

## Sources

### Primary (HIGH confidence — direct codebase read)
- `wargame_rl/wargame/envs/domain/los.py` — Bresenham, interior-only `is_blocking`, OOB handling (lines 16–77).
- `wargame_rl/wargame/envs/wargame.py` — `_make_is_blocking` (196–201), `has_line_of_sight_between_cells` (203–215), `iter_los_cells_between_cells` (217–221), `_battle` wiring (131–136).
- `wargame_rl/wargame/envs/domain/battle.py` / `battle_factory.py` / `battle_view.py` / `value_objects.py` — aggregate, factory, Protocol, `DeploymentZone` precedent.
- `wargame_rl/wargame/envs/types/config.py` — `blocking_mask` field + `normalize_blocking_mask`/`validate_blocking_mask_shape`; `ObjectiveConfig`; `_validate_entity_configs`; `deployment_zone` tuple.
- `wargame_rl/wargame/envs/env_components/shooting_masks.py` + `observation_builder.py` (135–144) — mask routes through `view.has_line_of_sight_between_cells`.
- `wargame_rl/wargame/envs/renders/human.py` — `_draw_debug_los_line` (440–472, traces only, no verdict), overlay drawing patterns.
- `wargame_rl/wargame/envs/state/snapshot.py` — confirmed **no** LOS computation in snapshot.
- `tests/test_los.py` — golden traces + interior-only contract to keep green.
- `.planning/phases/01-terrain-los-blocking/01-CONTEXT.md`, `.planning/REQUIREMENTS.md`, `.planning/ROADMAP.md`, `.planning/config.json`.
- `.cursor/rules/{python-conventions,gymnasium-env,testing}.mdc`, `docs/ddd-envs.md`.

### Secondary (HIGH for repo intent, partially superseded)
- `.planning/research/ARCHITECTURE.md`, `PITFALLS.md`, `SUMMARY.md` — seam/backward-compat/test guidance reused; wall-rasterisation content superseded by footprint LOS (CONTEXT.md).

### Canonical rules reference (from CONTEXT.md — for the LOS model)
- Wahapedia 10e Core Rules → Terrain Features → **Ruins** (FOOTPRINT + VISIBILITY); Chapter Approved 2025–26 → Terrain Layouts (footprint sizes, board scale). Source of D-01..D-06.

## Metadata

**Confidence breakdown:**
- Standard stack / integration points: **HIGH** — every seam and consumer read directly; no new deps.
- Architecture (endpoint-aware predicate, DDD wiring): **HIGH** — mirrors existing objectives/blocking_mask patterns 1:1.
- Pitfalls (symmetry, interior-only, mask↔resolution, renderer): **HIGH** — grounded in code + carried from prior repo research.
- Renderer overlay specifics (colours/alpha): **MEDIUM** — D-discretion; pygame test assertions kept light.

**Research date:** 2026-06-20
**Valid until:** ~2026-07-20 (stable; no fast-moving external deps). Re-verify only if `domain/los.py` or the LOS seam changes.
