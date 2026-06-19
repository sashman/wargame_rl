# Stack Research

**Domain:** Terrain & line-of-sight blocking for an existing RL tabletop wargame (v2.0 milestone)
**Researched:** 2026-06-19
**Confidence:** HIGH

## TL;DR — No New Dependencies

**No new libraries are needed for this milestone.** Footprint + wall terrain that blocks
Bresenham LOS and is encoded in the observation is entirely **domain logic + NumPy**, both
of which the project already has. The work is extending existing pure-Python/NumPy domain
services and the observation pipeline — not a stack change.

The integration surfaces already exist and are the only things to touch:

| Surface | File | Role for terrain |
|---------|------|------------------|
| Bresenham LOS | `wargame_rl/wargame/envs/domain/los.py` | Already traces interior cells via an injectable `is_blocking(x, y)` predicate — walls feed this |
| Blocking predicate | `WargameEnv._make_is_blocking` (`envs/wargame.py:196`) | Currently derives predicate from `config.blocking_mask`; walls become the new source |
| Config | `WargameEnvConfig.blocking_mask` (`envs/types/config.py:211`) | Existing terrain config slot + validators; add structured `terrain` field alongside |
| Observation struct | `WargameEnvObservation` (`envs/types/env_observation.py`) | Add terrain field + `size_*` accounting |
| Observation builder | `env_components/observation_builder.py` | Map `BattleView` terrain → observation |
| Tensor pipeline | `model/common/observation.py` | Encode terrain into tensors |
| Transformer net | `model/net.py` | New terrain token embedding (entity pattern) |

## Recommended Stack

### Core Technologies (all already present — keep as-is)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| NumPy | already pinned (`numpy` in `pyproject.toml`) | Wall/footprint grids, rasterization, vectorized observation features | Grid blocking is a 2D boolean array; rasterizing a thin wall segment is the same Bresenham the project already runs. No geometry lib needed. |
| Pydantic + pydantic-yaml | `pydantic-yaml>=1.6.0` | New `TerrainPieceConfig` / wall / footprint config models, YAML authoring, init-time validation | The project's established config pattern; `blocking_mask` already validated here. Mirror it. |
| Pure-Python Bresenham | in-repo (`domain/los.py`) | LOS ray + wall-segment rasterization | Single source of truth already exists, reused by rules/masks/renderer. Extend, don't duplicate. |
| Gymnasium / PyTorch / Lightning / Transformer | unchanged | Env API, training, entity-token network | Terrain is new observation content, not new infrastructure. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| (none) | — | — | **Nothing new is justified for this milestone.** |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| Ruff / mypy / pytest | existing lint/type/test gates | Add terrain-LOS unit tests next to `tests/test_los.py`; backward-compat test that no-terrain configs are byte-for-byte unchanged. |

## Installation

```bash
# No new packages. NumPy and Pydantic are already dependencies.
# (Existing) sync after pulling:
just dev-sync
```

## The Key Technical Decision: Cell-Edge vs Whole-Cell Wall Blocking

This is a **domain-modeling decision, not a stack decision** — both options are pure
Python/NumPy and require no library. Stack-relevant implications of each:

### Option A — Whole-cell blocking (mark wall cells opaque)
- **Maps directly onto the current code.** `has_line_of_sight` already calls
  `is_blocking(x, y)` per strictly-interior cell. A wall rasterized into the existing
  `blocking_mask`-style boolean grid works with **zero LOS signature change**.
- Rasterize each thin wall segment → cells using the in-repo `_bresenham_line` (so the
  wall and the LOS ray use the same discretization, avoiding "ray slips past wall" bugs).
- Cost: a "thin" wall occupies whole cells, so it is as thick as one grid cell. On a coarse
  grid this can over-block (a wall cell also sits "in front of" diagonal sightlines).
- Observation encoding is trivial: a `(H, W)` wall channel, or per-piece token features.

### Option B — Cell-edge segments (block the boundary between two adjacent cells)
- More faithful to "thin walls" and L-shaped pieces; a wall lives on the line *between*
  cells, so models in adjacent cells aren't themselves "in terrain."
- **Requires a new LOS predicate signature.** `is_blocking(x, y)` (point test) is
  insufficient; you need `crosses_blocked_edge((x0,y0) -> (x1,y1))` evaluated on each step
  of the ray between consecutive Bresenham cells. This is a small, additive pure-Python
  extension to `domain/los.py` (keep the existing point predicate path for `blocking_mask`
  backward-compat, add an optional edge predicate).
- Bresenham's diagonal steps complicate edge tests (a diagonal move crosses a corner, not a
  single edge) — needs a documented corner rule. Still pure Python, but more logic + tests.
- Encoding: edges are naturally `(H, W, 2)` (vertical + horizontal edge grids) or per-wall
  segment tokens.

### Recommendation
**Start with Option A (whole-cell), with walls authored as edge/segment data in config but
rasterized to a cell grid for LOS.** Rationale:
1. It reuses the existing `is_blocking(x, y)` path and `blocking_mask` plumbing with **no LOS
   API change** — smallest, safest brownfield change.
2. It keeps the LOS ray and wall on the same Bresenham discretization (consistency).
3. Authoring walls as segments (endpoints within the footprint) preserves the option to move
   to true edge-blocking later without changing the config schema — only the rasterization /
   LOS predicate changes.

Confidence: **MEDIUM** — this is a design trade-off, not a verifiable fact. Flag for the
planning phase to confirm against the desired tabletop fidelity. Either way, **no dependency
implication**.

## Observation Encoding — Stack Implications

The transformer is **entity-token based** (`net.py`: separate `nn.Linear` embeddings for
game / objectives / player / opponent tokens, concatenated). Terrain fits this in one of two
pure-NumPy ways — both no-dependency:

- **Entity tokens (recommended, fits the architecture):** emit one token per terrain piece
  (footprint rect bounds + wall segment endpoints, normalized like model locations). Add a
  `terrain` tensor + a `terrain_embedding` linear layer following the exact pattern used for
  opponents (which already defaults to "0 rows / no embedding" when absent — the backward-compat
  template to copy). Variable terrain count is handled by attention naturally.
- **Grid channel (simpler, less scalable):** flatten the wall mask `(H, W)` into game features.
  Cheap but couples observation width to board size and scales poorly; prefer tokens.

Either path is additive NumPy + one Linear layer. The established "add a new entity" checklist
(`.cursor/rules/pytorch-dqn.mdc`) is the exact procedure: obs struct → builder → `_observation_to_numpy`
→ `observation_to_tensor`/`_batch` → **both** networks → tests.

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| Hand-rolled NumPy/Bresenham wall rasterization | **Shapely** (geometry: segments, polygons, intersections) | Only if terrain becomes continuous (non-grid) polygons with true geometric LOS. On a discrete grid it is overkill, adds a C/GEOS dependency, and is *slower* than integer Bresenham for this use. Not justified now. |
| Reuse in-repo Bresenham | **scipy.ndimage** (line drawing, flood fill, binary masks) | Only if you later need flood-fill visibility / region labelling for "dense terrain" (a *deferred* milestone). SciPy is **not currently a dependency** — don't add it for footprint+wall LOS. |
| Boolean NumPy grid / per-piece tokens | **bitsets / specialized grid libs** (e.g. roguelike FOV libs like `tcod`) | Only for large maps needing precomputed symmetric FOV. Massive feature overlap with what exists; wrong abstraction for an RL env. Avoid. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Shapely | Pulls in GEOS native dependency; continuous-geometry power is unused on a discrete grid; slower than integer Bresenham here | Existing `domain/los.py` Bresenham + boolean wall grid |
| SciPy (`ndimage`) | Not a current dependency; not needed for segment LOS; only relevant to deferred dense-visibility/flood-fill features | NumPy boolean arrays; revisit if/when dense terrain milestone lands |
| `python-tcod` / roguelike FOV libs | Heavy game-engine FOV machinery; duplicates the single-source LOS service; foreign to the DDD env layout | Extend the existing domain LOS service |
| Any game engine (Unity ML-Agents, Godot RL) | Reiterating v1 research: massive overhead vs. dice/grid math | Gymnasium env as-is |
| A second/duplicate Bresenham for walls or rendering | Phase 3 explicitly chose a single LOS source reused by rules, masks, and renderer | Route walls through the one `has_line_of_sight` path |

## Stack Patterns by Variant

**If whole-cell blocking is chosen (recommended):**
- Reuse `is_blocking(x, y)` unchanged; build the wall boolean grid at config/episode init.
- Because the existing `blocking_mask` predicate is already wired, terrain becomes "another
  source ORed into the same mask" — minimal LOS surface change.

**If cell-edge blocking is chosen:**
- Add an optional edge predicate to `domain/los.py` (additive; keep the point predicate for
  `blocking_mask` backward-compat) and a documented diagonal/corner rule.
- Still pure Python; more logic and more tests, no new dependency.

**If terrain count is small and fixed per scenario:**
- Per-piece entity tokens in the transformer are clean and cheap.

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| NumPy (current) | Pydantic / pydantic-yaml 1.6+, PyTorch, Gymnasium 1.x | No version changes; terrain uses already-validated stack. No compatibility risk introduced. |

## Backward Compatibility (constraint from PROJECT.md)

- New `terrain` config field must default to **empty / None** so existing YAML behaves exactly
  as today (the no-op default pattern already used by `blocking_mask=None` and
  `number_of_opponent_models=0`).
- Observation additions must not change tensor shapes for no-terrain configs (mirror the
  opponent "0 rows / no embedding when absent" handling in `net.py` / `observation.py`).

## Sources

- In-repo code (HIGH): `wargame_rl/wargame/envs/domain/los.py`, `envs/wargame.py` (`_make_is_blocking`, `has_line_of_sight_between_cells`), `envs/types/config.py` (`blocking_mask` + validators), `env_components/observation_builder.py`, `types/env_observation.py`, `model/common/observation.py`, `model/net.py`, `pyproject.toml`.
- `.cursor/rules/pytorch-dqn.mdc` (HIGH) — "add a new entity" observation/tensor/network checklist.
- `docs/ddd-envs.md` (HIGH) — domain/BattleView extension boundaries.
- `.planning/research/v1-STACK.md` (HIGH) — prior conclusion (Bresenham + NumPy terrain grid, no new deps) reused, not redone.
- Cell-edge vs whole-cell trade-off (MEDIUM) — reasoning from the existing Bresenham point-predicate signature; not an external/version-sensitive fact. Confirm in planning.

---
*Stack research for: terrain + LOS blocking on a discrete grid RL wargame*
*Researched: 2026-06-19*
