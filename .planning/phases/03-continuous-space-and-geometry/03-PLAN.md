# Phase 03 — Continuous Space & Model Geometry

> Numbered as a phase, not a milestone: `v3.0` is already taken in `PROJECT.md` by
> *Advanced movement & deployment*. This work **precedes** it — advance and fall-back
> moves are distance budgets, which need continuous positions and bases underneath them
> to mean anything.

**Status:** ready to execute. Not started.
**Prototype:** `feature/continuous-positions-and-bases`, three commits off `0704272`
(`168f036`, `db4a4dd`, `3c7812e`) — 85 files, +5541/−631, 841 tests passing.

---

## What this is

The environment was a chessboard: `np.int32` cell positions, polar displacements rounded
to whole cells, models as dimensionless points, terrain as axis-aligned rectangles,
objectives as circles. This phase makes the board continuous and gives everything on
it real geometry.

A working prototype exists. **It was built to learn from, not to keep.** Everything below
is written with that prototype's results in hand, so each work package says both what to
build and — where the prototype got it wrong — what to do differently. Treat the branch as
a reference implementation to consult and selectively lift, not as a base to merge.

## Why bother

Two of the environment's quantisations were destroying information, not just approximating
it:

- `np.rint` on the displacement table collapsed **96 movement actions to 80 distinct
  outcomes**, and made a "speed 1" diagonal move travel 1.41 — 41% further than a speed 1
  orthogonal one.
- `.astype(int)` in the observation builder truncated the vector-to-objective — the single
  most informative feature the policy has — to whole units.

And the long-running cover question ("can the agent use terrain?") was never fairly asked:
until this work there was no cover rule at all, and terrain reached the network as a
bounding box, so an L-shaped ruin and a solid block were literally the same input.

## What the prototype settled

These do not need re-deciding.

| Question | Decision |
|---|---|
| Unit mapping | `inches_per_unit` scale, default 1.0. Coordinates in units; rules distances authored in inches. |
| Rules quantities | All derived from inches, resolved once at construction. Config values are overrides. |
| Base radius affects | Objective control · engagement · LOS occlusion and cover · movement collision |
| LOS | Sampled ray, keeping a blocking predicate rather than exact segment/rect intersection |
| Action space | Unchanged (16×6, 97 actions). Delete `np.rint` only. |
| Shapes | Footprints and objectives are polygons; objectives keep the disc as default |
| Walls inside ruins | Deferred. A footprint is an outline; a concave footprint is not a stand-in for a wall. |

## What the prototype proved works

Lift these approaches directly — they were measured, not assumed.

1. **The networks need no changes.** No embedding table, no positional encoding; positions
   already arrive as floats normalised by board half-size, and input dims derive from
   entity counts. Only the terrain token width changes.
2. **`RulesQuantities` resolved once at construction.** Runtime reads plain floats and
   never converts. Cost: nil.
3. **Python constants mirroring `docs/rules/constants.yaml`, pinned by a test.** No runtime
   file IO, no packaging problem, and the spec cannot drift from the code.
4. **The `norms_offset` trick for area objectives.** Measure to the area's edge and give it
   radius 0, and every downstream `norms_offset <= obj_radii` test works unchanged — no
   branch anywhere in the reward, VP or criteria layers.
5. **Vectorising across shapes, not just samples.** See WP-8.
6. **A denylist test over tracked *and untracked* files** for product references.

---

## Work packages

Ordered by dependency. WP-0 and WP-1 are independent of everything else and can ship on
their own.

### WP-0 — Delete the two quantisations

**Independent. Highest value per line in the whole phase. Do it first and alone.**

- `env_components/actions.py` — `self._displacements = raw`, deleting `np.rint(...).astype(int)`.
- `env_components/observation_builder.py` — drop `.astype(int)` / `dtype=int` on the
  objective deltas.

`n_actions` is unchanged, so checkpoints keep loading and the shooting head is untouched.
Measure before and after: distinct displacements should go 80 → 96, and speed-bin-0 moves
should all have length 1.0.

> **Ship this separately and train on it.** It is the one change here that improves the
> environment without changing the dynamics, so it is the only one whose effect on learning
> can be measured cleanly. Everything after this shifts the baselines.

### WP-1 — Rules specification and the IP guard

Already done well in `168f036`; **lift it wholesale**. `docs/rules/` (16 chapters,
`constants.yaml`, `implementation-status.md`), plus `tests/test_no_ip_references.py` and
`tests/test_rules_constants.py`.

Two things the prototype learned the hard way:

- Attribute docstrings after a module-level assignment trip `check-docstring-first`. Use
  comments.
- The pre-commit `mypy` hook runs in its own isolated env — `types-PyYAML` has to be in
  `additional_dependencies`, not just the dev group.

### WP-2 — Scale and rules quantities

`domain/rules_constants.py` (inch values), `types/geometry.py`-adjacent `Scale`,
`domain/rules_quantities.py` with `resolve_rules_quantities(config)`.

Convention to state once and hold to: **coordinates and anything that generates one are in
units; rules distances are in inches.**

Config gains `inches_per_unit`, `base_radius`, `engagement_range`, `los_sample_step`; the
existing distance fields are reinterpreted as inches. At the default scale every shipped
config is numerically unchanged — that is the safety property, and it is worth preserving.

**Expect the defaults to move:** engagement 1 → 2, objective radius 1 → 3, group distance
10.0 → 9.0, and weapon range only where a config omits the override.

### WP-3 — Continuous positions

Float64 locations, float32 Gym `Box`es with `high=board_width` (not `-1`), float sampling
in placement, drop the `int32` casts in `load_state`.

Keep `board_width`/`board_height` as `int`. A whole-number board extent is harmless and it
keeps the renderer's `range()` loops and the mask-shape validators working. The prototype
tried floating them and reverted.

> **Trap — the corner-inclusive convention.** A footprint authored `(5,5,5,5)` means *one
> cell*; read literally as a continuous rectangle it has **zero area**, and `(27,8,33,16)`
> shrinks from 7×9 to 6×8. Convert at the config boundary (`from_cell_rect` pushes the far
> corner out by one). The same `-1` lurks in mirroring: reflect about `width - x`, not
> `width - 1 - x`. Nothing fails when you get this wrong — the terrain just quietly gets
> smaller.

### WP-4 — Model bases

`base_radius` on the model. Objective range measured from the base *edge*; engagement
measured base-to-base; the board clamp inset by a radius; radius-aware non-overlapping
placement; the renderer drawing circles at the real radius and dropping the cell-centre
`+ 0.5` at nine sites.

> **Constraint discovered:** a base must fit in its deployment zone. A 5×5 board's zone is
> 1 unit wide and a 32 mm base is 1.26 across, so small test and demo configs have to grow.
> Fail loudly at placement with the numbers in the message.

### WP-5 — Units as a real entity — **NEW, not in the prototype**

**This is the largest correction the prototype produced, and it gates WP-8.**

The rules say a model ignores others *in its own unit* and *in the target's unit* when
tracing sight. The prototype used `group_id` as the unit proxy — but the default config
assigns every model its own `group_id`, so a squad occluded itself and "open ground is
fully exposed" became false.

`group_id` is also overloaded: it drives cohesion rewards, placement clustering and
baseline squad assignment. Give units a real identity before anything depends on unit
membership for correctness.

Scope check while you are here: `WargameModel`'s own docstring conflates model and unit,
there is no `Unit` class, and `docs/rules/` is written throughout in terms of units. This
work package is where that gap closes.

### WP-6 — Movement collision

Bases cannot overlap. Swept-circle movement: enemy bases block the path, friendly bases may
be passed through but not ended on, and the model backs off to the last clear point.

Two things the prototype got wrong, both caught by tests:

1. **Backing off alone makes squads queue.** Everything aiming at an objective centre stops
   behind whoever arrived first, forming a radial line. `greedy_nearest` fell to reaching
   1 model in 3. Adding a tangential slide recovers most of it.
2. **Sliding is a movement exploit unless budgeted.** It let a model travel 6.046 against a
   Move of 6.0, because the tangential step was added on top of distance already covered.

> **And the correction the prototype could not make:** one-pass sliding is still not
> enough. `greedy_nearest` plateaus at ~0.71 regardless of episode length — models orbit
> the cluster indefinitely. The objective disc holds ~30 bases, so this is a **pathfinding**
> limit, not a capacity one. **The fix belongs on the policy side: assign each model a
> distinct target slot around the objective.** Collision response cannot substitute for it.
> Do this in WP-6, not later — every baseline measured before it is misleading.

Resolution is sequential in model index order, which carries a documented right-of-way
bias. That is the price of determinism, and the seeded env and its tests depend on it.

### WP-7 — Polygon geometry

`Polygon` with vectorised point-in-polygon (crossing number, so concave outlines work),
area, centroid, distance-to-point, intersection, mirroring, and padding to a vertex budget.

**Put it in `types/`, not `domain/`.** `config.py` needs it for validation and cannot import
from `domain/` without inverting the dependency direction. The prototype put it in `domain/`
and had to move it.

Two boundary decisions worth copying:

- **`contains()` is boundary-inclusive; `contains_points()` is not.** Endpoint membership
  decides whether a model is standing *in* a piece and so can see out of it — a model
  placed exactly on a corner by a fixed config would otherwise be blocked by its own cover.
  Sample membership is measure-zero and on the hot path, so it stays cheap.
- **Touching is not overlapping.** Adjacent cell rectangles share an edge once continuous,
  and treating that as an overlap rejects layouts that are plainly fine.

### WP-8 — Line of sight, occlusion and cover

Sampled ray replacing Bresenham. Model bases occlude, ignoring the observer's unit and the
target's (needs WP-5). Three-state visibility — hidden / visible / fully visible — from
three rays: centre-to-centre plus the two outer tangents. Cover is the middle state, and
worsens the attack's hit target by 1.

> **Performance is the whole game here, and the lesson is specific: vectorise across
> shapes, not just across samples.** Per-piece looping cost **70.2 ms/step against 24.2** —
> a 3× regression — because line of sight runs hundreds of queries per phase and the loop
> turned each into dozens of tiny numpy calls whose overhead dwarfed the arithmetic. Pad
> every outline to a common vertex count and test rays × samples × polygons × edges in one
> pass: **30.0 ms**. Padding is free because repeated vertices make zero-length edges,
> which never straddle a sample.

Measured costs, for judging any regression:

| | ms/step (25v25, 29 pieces) |
|---|---|
| Bresenham, no occlusion | 22.2 |
| Rectangles, sampled ray + occlusion | 24.2 |
| Polygons, per-piece loop | 70.2 |
| Polygons, one vectorised pass | **30.0** |

Other traps:

- **A model blocks its own sight.** The observer's base sits on the start of every ray it
  casts. Generalise terrain's see-out rule to discs: drop any occluder covering either
  endpoint.
- **Thin features leak.** A blocker narrower than the sample step falls between samples.
  Reject them at config load.
- **The legacy `blocking_mask` is opaque matter, not shelter.** The see-out endpoint filter
  must not apply to it, or a model standing near a masked cell sees through it.
- **Cover is only wired into the player's shooting path in the prototype.** The opponent's
  resolution does not check it. Close that here.

### WP-9 — Polygon terrain and objectives

`TerrainPieceConfig` takes a rectangle or an outline, exactly one. `ObjectiveConfig` gains
an optional `polygon`, making it the rules' terrain objective — the area *is* the
objective. Random generation produces convex n-gons; mirroring reflects vertices, and the
centre piece of an odd mirrored layout is built symmetric **by construction** (sample half
the vertices, mirror them) rather than by hulling a shape with its own reflection, which
doubles the vertex count and overflows the observation budget.

An area objective is not *placed* — its outline is its position, and its `location` is the
centroid so anything steering toward an objective still has a point to aim at.

> **Trap — calibrations that assumed rectangles break silently, in both directions.**
> Convex outlines *hide less* board than the rectangles they replace (cells-hidden 0.198 →
> 0.174, needing size 3–7 → 4–8), but they *pack tighter*, so a packing validator that
> assumes a piece fills its size box starts rejecting profiles the sampler places on every
> seed. Re-derive both against the generator, do not port the numbers.

### WP-10 — Terrain observation

Terrain reaches the network as padded outline vertices plus a vertex count, not a bounding
box. `TERRAIN_FEATURE_DIM` 4 → `2 · V_max + 1`.

This is the first encoding that lets the policy tell an outline from its bounding box, and
therefore the first honest test of whether the agent can use terrain — the whole
cover-experiment line of work was run against an input that could not express the question.
It invalidates checkpoints, which by this point in the sequence is already true.

### WP-11 — Schema, scripts, docs

Snapshot and event schemas go float, gain `base_radius`, and bump the version. The delta
encoder compares locations by equality, so under floats every model emits a delta every
step and compression collapses — it needs a tolerance. `state/analysis.py` computes
oscillation by exact tuple equality and would silently report zero forever; quantise it for
that metric or drop it. `scripts/measure_terrain.py` is fully raster and has to be ported
before WP-12 can use it.

> **Trap — making an event more frequent finds the bugs conditioned on it.** Enlarging the
> objective radius from 1 to 3 surfaced a **pre-existing** bug: `load_state` zeroed the VP
> deltas instead of restoring them, so a snapshot round-trip was lossy on any step that
> scored. It presented as an intermittent test failure that passed in isolation, which
> reads like flakiness and is not. Expect more of this shape.

### WP-12 — Re-tune and re-measure

Re-tune the terrain profile with `measure-terrain` against the actual generator. Re-measure
every baseline. Update `implementation-status.md` — it is the doc most likely to mislead
once it goes stale.

---

## Re-measurement discipline

**Every baseline in the repo is invalidated by this phase, more than once along the
way.** Measured on `25v25_cover_control.yaml`, six seeds:

| stage | `random` | `squad_march` | `squad_march_shoot` |
|---|---:|---:|---:|
| before (recorded in CLAUDE.md) | — | — | 0.45 |
| bases + continuous, rect terrain | 0.00 / −190.0 | 0.50 / +4.2 | 1.00 / +99.2 |
| polygon terrain | 0.00 / −209.2 | **0.83** / +50.0 | 1.00 / +107.5 |

Two readings worth carrying forward:

- The bar moved **0.45 → 1.00** on the same config and the same weapon range. Nothing
  measured before this milestone transfers.
- The **movement-only** policy is the sensitive one. It went 0.50 → 0.83 on polygon terrain
  (rounded outlines are easier to walk around) and +4.2 → −41.7 when weapon range doubled
  (longer exposure before arrival). A policy that never shoots lives or dies on reaching
  the objective, which makes it the sharper instrument for detecting a geometry change.

Read `vp_margin`, not win rate, for arm-to-arm comparison; within-arm seed spread on win
rate is 6–7pp. Two seeds per arm minimum.

---

## Deferred, deliberately

- **Walls inside ruins.** The L- and U-shaped structures that break sight *within* a piece.
  A footprint is an outline; nothing in the geometry assumes convexity, so whichever way
  walls are modelled the outline will not be in the way.
- **Terrain categories.** One category exists (an LOS-blocking outline); the rules define
  exposed, light and dense with different movement and cover behaviour.
- **Terrain and movement.** Terrain still does not affect movement at all.
- **Coherency enforcement.** The 2" nearest-neighbour rule has no consumer, and nothing
  destroys models for breaking coherency.
- **Advance and fall-back moves.** `advanced_this_turn` is read by the shooting mask and
  never set.
- **Board height.** No vertical dimension, so elevated fire, the 3" solid threshold and
  vertical movement have no analogue.

## Open question

**Whether to derive weapon range from the rules.** The rules value is 24"; the shipped
configs override to 12. The prototype measured both: the scripted bar degrades gracefully
(1.00 → 0.83) but the movement-only policy collapses (+4.2 → −41.7), and the terrain
profile was tuned to break firing lanes at 12. `reports/2026-08-05` separately found a
*learned* agent collapsing to 6.8% win rate at doubled range. Decide deliberately, per
scenario, and re-tune terrain with the range rather than after it.
