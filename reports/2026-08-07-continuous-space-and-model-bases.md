# Continuous space and model bases — prototype findings

**Date:** 2026-08-07
**Status:** prototype. Built to learn what changes when the board stops being a
chessboard, not as the implementation to keep.

## What was built

The environment was a lattice: `np.int32` positions, polar displacements rounded to
whole cells, and models as dimensionless points. It now has continuous float positions,
an inch ↔ board-unit scale, and models that occupy space as circular bases which cannot
overlap. Line of sight is a sampled ray that model bases can block, which makes cover
real for the first time.

Everything below was measured, not assumed.

---

## The findings that matter

### 1. The old action space was throwing away a fifth of itself

`np.rint` on the displacement table collapsed **96 movement actions to 80 distinct
outcomes**, and stretched the diagonals: at speed bin 0 the sixteen directions came out
with lengths `1.0` and `1.414`, so a "speed 1" diagonal move actually travelled 41%
further than a speed 1 orthogonal one.

Deleting the rounding is a one-line change and restores all 96. This is the single
cheapest improvement found, and it does not depend on anything else in this prototype.

### 2. Naive collision response makes squads orbit, not queue

The first implementation stopped a model dead when its destination was occupied. That
looks reasonable and is badly wrong: every model aiming at an objective centre stops
behind whoever arrived first, so a squad forms a **radial queue** pointing at the
objective instead of spreading around it. `greedy_nearest` fell from reaching every
model in to reaching **1 in 3**.

Adding a tangential slide — spend the movement you gave up by stopping, running along
the obstruction rather than into it — recovers most of it. But not all, and the residue
is the interesting part:

| baseline | fraction of models on an objective |
|---|---|
| `squad_march` (spreads across a frontage) | **1.00** |
| `split_evenly` | 0.88 |
| `greedy_nearest` (all models at one point) | **0.71**, and it plateaus there |

`greedy_nearest` does not improve with more rounds — 12, 20, 30 and 50 all give 0.83
before the slide-budget fix and ~0.71 after. The objective disc has room for roughly
**thirty** of these bases, so this is not a capacity limit. Models slide tangentially
around the cluster and orbit it indefinitely.

**For the real implementation:** one-pass tangential sliding is not enough. A policy
that converges every model on a single point needs *slot assignment* — distinct target
positions around the objective — which is what a human player does without thinking
about it. Collision response alone cannot substitute for it.

### 3. Sliding is a movement exploit unless you budget it

The property test caught this immediately: sliding let a model travel **6.046** against
a Move of 6.0, because the tangential step was added on top of the distance already
covered rather than spending what remained. Any collision response that redirects
movement has to debit the same budget.

### 4. Model occlusion is nearly free — the cost was never the geometry

The worry was that sampled-ray line of sight with model bases as occluders would be an
order of magnitude slower than Bresenham. On 25v25 with 29 terrain pieces:

| | ms/step |
|---|---|
| Bresenham, no occlusion | 22.23 |
| Sampled ray, 3 rays per query, model occlusion | **24.23** |

**8% slower**, for three rays per query instead of one *and* a 50-disc occlusion test.
The win is vectorising per ray — building the sample points as one array and testing
every blocker at once — rather than looping in Python. Range pruning before the line of
sight query does the rest.

### 5. A model that occupies space blocks its own line of sight

Obvious in hindsight, invisible in advance. The observer's own base sits on the start
of every ray it casts, so every model was blind. The fix generalises the rule terrain
already had — a model can see out of the ruin it stands in — to discs: drop any
occluder covering either endpoint.

### 6. Corner-inclusive rectangles silently shrink every terrain piece

A footprint authored as `(5,5,5,5)` means *one cell*. Read literally as a continuous
rectangle it has **zero area**; `(27,8,33,16)` goes from 7×9 cells to 6×8 units. Every
piece loses a unit on each axis, and the tuned terrain profile changes underneath you
with nothing failing.

Converting at the config boundary (`Footprint.from_cell_rect` pushes the far corner out
by one) preserves the authored area exactly. The related trap: mirroring reflects about
`width - x`, not `width - 1 - x` — the `-1` is a last-cell-index convention that has no
meaning once coordinates are continuous.

### 7. Small boards become illegal

A 5×5 board has a deployment zone 1 unit wide. A 32 mm base is 1.26 units across, so it
does not fit, and placement now fails loudly. This is correct — a 5-inch board is
smaller than a single model's move — but it means any test or demo config built for
speed rather than realism has to grow.

### 8. Every baseline number in the repo is now wrong

This is the one with the widest blast radius. On `25v25_cover_control.yaml`, six seeds:

| weapon range | policy | win rate | vp_margin | exposure | firepower ratio |
|---:|---|---:|---:|---:|---:|
| 12" | `random` | 0.00 | −190.0 | 0.034 | 0.808 |
| 12" | `squad_march` | 0.50 | +4.2 | 0.561 | 1.176 |
| 12" | `squad_march_shoot` | **1.00** | +99.2 | 0.339 | 0.976 |
| 24" | `random` | 0.00 | −202.5 | 0.105 | 0.672 |
| 24" | `squad_march` | 0.50 | −41.7 | 0.694 | 0.914 |
| 24" | `squad_march_shoot` | **0.83** | +80.0 | 0.480 | 1.081 |

`CLAUDE.md` records the bar on this config as `squad_march_shoot` at **0.45**. It is now
**1.00** at the same weapon range. The dynamics changed enough — bases, base-to-base
engagement at 2", cover, exact displacements, collision — that no previously measured
score transfers. **Do not compare anything to a pre-2026-08-07 baseline.**

### 9. Doubling weapon range costs less than expected, and punishes standing still

`reports/2026-08-05-stochastic-terrain-and-cover.md` found that doubling weapon range
collapsed a *learned* agent's win rate to 6.8%. The scripted bar degrades far more
gracefully: 1.00 → 0.83, vp_margin +99 → +80.

The sharper effect is on the policy that does not shoot. `squad_march` goes **+4.2 →
−41.7** — at 24" you are under fire for far longer before you arrive, so declining to
return fire stops being survivable. Exposure rises with range for every policy
(`squad_march_shoot` 0.339 → 0.480), which is the mechanism.

Note the configs set `range: 12` explicitly, and that is now read as 12 **inches**. The
rules-correct 24" only applies where a config omits the override, so this remains a
deliberate per-scenario choice rather than something the refactor forced.

---

## What this says for the real implementation

1. **Keep:** the inch ↔ unit scale as a single resolved-at-startup object. It cost
   nothing at runtime and made every rules quantity traceable to the specification.
2. **Keep:** vectorised sampled-ray visibility. The three-state answer (hidden /
   visible / fully visible) is what cover needs, and it is affordable.
3. **Redesign:** collision response. Sliding is a patch. Convergent objectives need
   target-slot assignment on the policy side; the physics cannot fix a pathfinding
   problem.
4. **Redesign:** `group_id` as the unit proxy. The rules say a model ignores others in
   its own unit when tracing sight, but the default config gives every model its own
   group, so squads occlude themselves. Units need to be a real entity.
5. **Watch:** the corner-inclusive convention. It is the one change here that is silent
   in every direction — no exception, no failing test, just terrain a unit smaller than
   the config asked for.

## Reproducing

```
just test                      # 816 passing
just lint
uv run pytest tests/test_model_bases.py     # the non-overlap property tests
```

Step-time and baseline numbers above come from `25v25_cover_control.yaml` with 6 seeds;
the measurement scripts are inline in this report's git history rather than checked in,
since the prototype is not the thing being kept.
