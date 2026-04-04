# Phase 3: Line of Sight Service - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in `03-CONTEXT.md` — this log preserves the alternatives considered.

**Date:** 2026-04-04
**Phase:** 3-line-of-sight-service
**Areas discussed:** Blocking in v1, Public API shape, Grid trace semantics, Model occlusion / renderer hooks

---

## 1. Blocking in v1 (before terrain)

| Option | Description | Selected |
|--------|-------------|----------|
| A | Injectable `is_blocking(x,y)` + optional config-backed mask; default all clear | ✓ |
| B | Tests-only blocking; no config hook until terrain v2 | |
| C | Hard-coded global board state in domain | |

**User's choice:** Discuss area **1** selected in multi-select `1,2,3,4`; facilitator recorded **A** per prior session recommendation and backward-compat requirements.

**Notes:** Optional YAML/config field keeps production boards unchanged; tests use explicit predicates.

---

## 2. Public API shape

| Option | Description | Selected |
|--------|-------------|----------|
| A | Pure functions in `domain/los.py` + env wires `is_blocking` and dimensions | ✓ |
| B | LOS only as methods on `Battle` aggregate | |
| C | Model-index-first API in Phase 3 | |

**User's choice:** Area **2** selected; **A** recorded — matches DDD and Phase 4 mapping from indices to cells later.

**Notes:** `has_line_of_sight` + `iter_los_cells` (names discretionary) for render parity.

---

## 3. Grid trace semantics (Bresenham)

| Option | Description | Selected |
|--------|-------------|----------|
| A | Integer Bresenham; interior cells only for blocking; exclude shooter and target cells | ✓ |
| B | Include target cell in blocking checks | |
| C | Same-cell query undefined / false | |

**User's choice:** Area **3** selected; **A** recorded with same-cell → **True** (D-09).

**Notes:** Diagonals follow standard Bresenham; document reference in module docstring.

---

## 4. Model occlusion & renderer hooks

| Option | Description | Selected |
|--------|-------------|----------|
| A | Terrain/mask blocking only in Phase 3; models do not block | ✓ |
| B | Living models on intervening cells block LOS | |
| C | Boolean LOS only; no cell iterator | |

**User's choice:** Area **4** selected; **A** recorded; shared cell trace for human render (**not C**).

**Notes:** Composite blocking deferred to a future phase.

---

## Claude's Discretion

- Mask storage layout (`height×width` vs `width×height`) and Pydantic field naming left to implementation.

## Deferred Ideas

- Model occupancy as LOS blocker — future phase.
- Full terrain v2 — separate milestone.
