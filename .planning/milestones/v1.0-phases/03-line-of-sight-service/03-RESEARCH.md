# Phase 3 — Technical Research: Line of Sight Service

**Date:** 2026-04-04
**Question:** What do we need to know to plan LOS on the discrete grid with Bresenham + injectable blocking?

---

## User Constraints

*(Verbatim scope from `03-CONTEXT.md` — planner MUST NOT contradict.)*

- **D-01–D-03:** Blocking via **`is_blocking(x, y) -> bool`** passed into queries; optional **config/YAML** mask defaulting to **no blocking**; tests use explicit predicates/fixtures.
- **D-04–D-07:** Core in **`wargame_rl/wargame/envs/domain/los.py`** (name may vary), **pure functions**, **no Gym imports**. APIs: **`has_line_of_sight`**, **`iter_los_cells`** (same trace). Callers pass **width, height, is_blocking**. No model-index API in Phase 3.
- **D-08–D-12:** Integer **Bresenham**; **`location[0]=x`, `location[1]=y`**; same cell → **True**; blocking checks **strict interior only** (exclude shooter and target cells); OOB → **`False`**; document algorithm in module docstring.
- **D-13–D-14:** **No model occlusion** in Phase 3.
- **D-15:** Renderers use **`iter_los_cells`** from the **same module** as bool LOS.
- **Deferred:** LOS-03 (masks), full terrain v2, indirect fire, occupancy blocking.

---

## Project Constraints (from `.cursor/rules/`)

- **DDD:** Domain under `domain/`; no imports from `env_components` / `reward` / `renders` in domain modules (`docs/ddd-envs.md`).
- **Config:** New fields **default to no-op** so existing YAML unchanged.
- **Tooling:** `just validate` before ship; ruff + mypy strict + pytest.
- **Types:** Full type hints on public APIs.

---

## Standard Stack

| Area | Choice | Confidence |
|------|--------|------------|
| Language | Pure Python 3.13 in `domain/los.py` | HIGH |
| Numerics | Optional `numpy` only if mask from config is ndarray — **prefer** closure over mask without requiring numpy inside `has_line_of_sight` core | HIGH |
| Tests | `pytest`, table-driven cases | HIGH |

---

## Architecture Patterns

1. **`BlockingPredicate`**: `Callable[[int, int], bool]` — single abstraction for tests and production mask closure.
2. **Trace iterator first or shared helper**: Implement **`_cells_along_ray(x0,y0,x1,y1)`** (generator or list) used by both `has_line_of_sight` (scan interior subset) and `iter_los_cells` (full ordered path) so render and logic cannot diverge.
3. **Bounds**: Before tracing, if either endpoint not in `[0, width) × [0, height)`, return **`False`** / empty iterator per CONTEXT D-11.
4. **Interior cells**: From full trace `[(x0,y0), ..., (x1,y1)]`, interior = `trace[1:-1]` when `len(trace) >= 2`; when same cell, trace length 1 → **no interior** → **True**.

---

## Bresenham on Square Grid

- Use the **integer 2D Bresenham line** algorithm (classic formulation: error-term loop over dominant axis). **Confidence HIGH** for determinism and wide reference material.
- **Pitfall:** “Supercover” vs standard Bresenham — use **one** and document; standard Bresenham is typical for grid LOS in games.
- **Pitfall:** Off-by-one on which cells are “between” shooter and target — locked by CONTEXT **interior only**.

---

## Don't Hand-Roll

- Do **not** duplicate LOS in `renders/` or `env_components/` — only **`domain/los.py`** (plus env wiring that **calls** it).
- Do **not** add shooting or `ActionRegistry` LOS masking in this phase.

---

## Common Pitfalls

| Pitfall | Mitigation |
|---------|------------|
| Mask axis order `(y,x)` vs `(x,y)` | Document indexing; match `model.location` |
| Iterator vs list divergence | Single `_cells_along_ray` source |
| Forgetting OOB | Explicit guard before Bresenham |
| Checking endpoints as blockers | Only `is_blocking` on **interior** |

---

## Code Examples (sketch — not prescriptive)

```python
def iter_los_cells(
    x0: int, y0: int, x1: int, y1: int,
    width: int, height: int,
) -> list[tuple[int, int]]: ...

def has_line_of_sight(
    x0: int, y0: int, x1: int, y1: int,
    width: int, height: int,
    is_blocking: Callable[[int, int], bool],
) -> bool:
    cells = iter_los_cells(x0, y0, x1, y1, width, height)
    if len(cells) <= 1:
        return True
    for x, y in cells[1:-1]:
        if is_blocking(x, y):
            return False
    return True
```

*(Planner: align with actual trace — if `iter_los_cells` returns full inclusive segment, interior slice is `cells[1:-1]` when endpoints differ.)*

---

## Validation Architecture

**Purpose:** Execution sampling and Nyquist Dimension 8 (automated verification map).

| Dimension | How this phase satisfies |
|-----------|---------------------------|
| Unit | `tests/test_los.py` — known boards, blocking lambdas, golden traces |
| Integration | Optional: env loads YAML with small mask; `has_line_of_sight` via env helper |
| Regression | Same-cell, clear line, blocked mid-ray, diagonal, OOB |

**Quick command:** `uv run pytest tests/test_los.py -q`
**Full suite:** `just test` or `just validate`

**Wave 0:** Not required — pytest already present.

---

## RESEARCH COMPLETE

Findings are prescriptive enough for a single executable plan: **`domain/los.py` + tests + optional config mask + env wiring + optional render debug using `iter_los_cells`**.
