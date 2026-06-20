# Phase 1: Terrain in the Simulation (LOS-Blocking) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-20
**Phase:** 01-terrain-los-blocking
**Areas discussed:** Authoring schema, Diagonal LOS at corners, Placement constraints, Demo & rendering, LOS model (canonical 10e Ruins)

---

## Authoring schema

| Option | Description | Selected |
|--------|-------------|----------|
| Footprint `[x, y, width, height]` | origin + size (matches deployment_zone) | |
| Footprint `[x0, y0, x1, y1]` | two opposite corners | ✓ |
| Walls as straight segments `[x0,y0,x1,y1]` | L = two segments | |
| Walls as polyline of points | one L = one polyline | ✓ (later superseded — walls deferred) |
| Absolute board coords | consistent with models/objectives | ✓ |
| Footprint-relative coords | relative to footprint origin | |

**Notes:** Walls were initially specced (polyline) but later removed from scope once the LOS model
was reset to the canonical footprint-based 10e Ruins rule. Footprint stays `[x0,y0,x1,y1]` absolute.

---

## LOS model (the pivotal decision)

The user directed a return to the source 40k 10e Ruins rules after we had layered conflicting
assumptions ("walls block, footprint no-op" → a hybrid → reset).

| Option | Description | Selected |
|--------|-------------|----------|
| Walls block LOS, footprint no-op | thin L-walls rasterised to blocking cells | |
| Hybrid | walls block + footprint see-in/out exception | |
| **Canonical 10e Ruins (footprint-based)** | footprint blocks when it sits between two outside models; see-into & see-out exceptions; walls have no LOS role | ✓ |
| Footprint-based + drop walls entirely | | partial — walls deferred, not dropped |

**User's choice:** Canonical footprint-based LOS ("option A, but lets revisit walls later").
**Notes:** Grounded in Wahapedia core rules (Terrain Features → Ruins, VISIBILITY paragraph) and the
Chapter Approved Terrain Layouts ("natural abstraction of line of sight within the rules for Ruins").
Walls deferred to a later milestone.

---

## Diagonal LOS at wall corners

**Status:** Rendered moot. This only mattered under wall-based LOS; with footprint-based LOS there
are no thin walls and no corner-pinhole cases. Not decided / not needed.

---

## Placement constraints

| Option | Description | Selected |
|--------|-------------|----------|
| Overlap deployment zones | models can deploy in/near ruins | ✓ |
| Overlap objectives | objectives inside ruins | ✓ |
| Models spawn on footprint cells | | ✓ |
| Disallow footprints overlapping each other | keep pieces distinct | ✓ |

**User's choice:** Allow overlap with deployment zones, objectives, spawn cells; footprints must not
overlap one another.

---

## Demo scenario

| Option | Description | Selected |
|--------|-------------|----------|
| New example config: 2 deployment zones + 1–2 ruins between them | clear LOS demo | ✓ |
| Add terrain to an existing config | | |
| Minimal one-ruin config | | |

**User's choice:** New example config with ruins blocking cross-board sightlines.

---

## Rendering

| Option | Description | Selected |
|--------|-------------|----------|
| Translucent filled rectangle + outline + label; debug LOS coloured by verdict | | ✓ |
| Outline only | | |
| Claude decides | | |

**User's choice:** Translucent filled footprint + verdict-coloured debug LOS line.

## Claude's Discretion

- Footprint "passes through" cell test, endpoint canonicalisation for symmetry, render colours/alpha,
  rectangles-vs-precomputed-grid representation.

## Deferred Ideas

- Walls (rendering/movement/finer LOS), cover, dense visibility, difficult/impassable movement,
  elevation/Plunging Fire, board templates, variable base sizes; terrain-in-observation is Phase 2.
