# The eval tables were traced by hand, and the tracing was the error

**2026-08-20.** `configs/evaluation/maps/` is now generated from the public
layout Data API by `just fetch-maps`. The 45 tables are the *same* 45 layouts
they always were — the hand-tracing that produced them is what changed.

**This is a re-derivation with provenance, not a new distribution.** Establishing
that was the first job, because it decides whether the change is a tidy-up or a
scenario swap. It is closer to the first than expected, and further than is
comfortable.

---

## 1. They are the same 45 layouts

Matched by axis-aligned piece bounds, every one of our 45 tables has a distinct
near-exact counterpart among the API's 45 `strike-force` layouts — a bijection.
The handful of rows where the second-best match is close are pairs of genuinely
near-identical *source* layouts, the same terrain reused for two mission slots.

So the numbering could be preserved, and it was. `maps_heldout` is still the
same nine tables, which is the only reason any before/after comparison here
means anything.

## 2. Three conventions the API does not document, settled by measurement

**The local frame is centred, not corner-relative.** Outline points are
piece-local and centred on the footprint origin, so a piece rotates about its own
middle and is then translated. The test is decisive rather than aesthetic: under
the centred reading all **720 pieces land on the board with zero overhang**,
while the corner reading throws **70 of them off the edge by up to 3.75"**.

**Simplification is mandatory.** Every source outline carries **167 to 348
vertices** against `TERRAIN_VERTEX_BUDGET = 8`. That is not a limitation to work
around; it is the explanation for why the hand-traced tables were quads.

**Douglas-Peucker is the method, chosen on numbers.** Measured over all 720
pieces against their true silhouette area:

| method | median | p90 | worst |
|---|---|---|---|
| footprint rectangle (4v) | 0.964 | 1.592 | **1.592** |
| convex hull → 8v | 1.068 | 1.160 | 1.160 |
| **Douglas-Peucker → 8v** | **1.027** | 1.078 | **1.078** |

The rectangle's tail is the disqualifier: an angled or L-shaped ruin blocks half
again as much board as it should. The hull fills in every concave bay.

## 3. An objective is a terrain piece

The current rules have no free-standing objective markers to stand near; the
marker says *which ruin* is fought over. So a marker resolves to that piece's
outline, and the piece is the objective.

Measured across the pool: **146 of 225 markers sit inside a piece**, 71 more
within the 3" control range. **Eight sit further** — at most 5.01" — and stay
discs on open ground. Those eight are deliberate, not bad data: **four are the
exact board centre**, and the distance 3.00" recurs to the inch, which is a
marker placed exactly one control range off a ruin.

Two markers on one ruin would collapse to a single objective, because that ground
is held once. **On this pool that never fires.** Every table carries five.

That 146 / 71 / 8 split is **identical against the dense source outlines and the
simplified 8-vertex ones** — not one marker changes category. It is the sharpest
evidence available that the vertex budget costs nothing load-bearing: the
simplification is invisible to the only question the geometry is asked.

### What the tracing had actually done

The old files' own comment said each objective was "the outline of the ruin the
layout puts a marker on". It was not. On `table_01`, **two of the layout's five
markers land inside an objective the file declares**; the rest sit elsewhere, and
the file's own comments name them `home`, `no man's land` and `centre` — chosen
by eye for board symmetry. The tables also carried **six** objectives on 27 of
45, which is the previous edition's mission count.

## 4. Two independent checks nobody designed for

Both were computed after the fact, and both came out exactly right — which is the
strongest evidence the pipeline is faithful.

- **The zone split is exactly 75 / 75 / 75** across player zone, middle and
  opponent zone. (It was 82/82/82 over 246 objectives before.)
- **The tables are point-symmetric to the measurement floor**: a table sits a
  median of **0.00"** from its own 180-degree rotation, worst 0.71". The
  hand-traced ones sat 1.7" out and at worst 3.9". *That asymmetry was tracing
  error.*

The second has a consequence: `map_pool.mirror` is **exactly 2x, not "about
2x"**. If `rot180` is the identity and it commutes with the x reflection, then
`flip_y` IS `flip_x` and `flip_xy` IS the original. Four orientations, two
distinct boards.

## 5. What changed on the board

| | traced | generated |
|---|---|---|
| pieces per table | 15 or 16 | **16** |
| outline | quads | 8-vertex silhouettes |
| objectives | 5 or 6, chosen by eye | **5**, resolved from markers |
| terrain area | 23.5% of board | 24.8% |
| own-zone/middle/opponent | 82/82/82 | 75/75/75 |
| distance from own rot180 | 1.7" median | **0.00"** |

**82% of objectives moved less than 3"; 18% moved further, up to 13.7".** That is
the part that voids baselines.

**The observation width did not change.** `objective_budget` stays 6 and
`terrain_budget` 16, so every checkpoint still loads and paired arms still pair.
Lowering the objective budget to 5 would change the tensor width and orphan every
checkpoint in `checkpoints/` — it must not be "tidied".

## 6. What it cost, measured

Four scripted policies on the golden training config, **all 45 tables**, n=30
each, old geometry against new, paired by table. All 45 are admissible here for
the same reason the joint decode was measured on all of them: **a scripted policy
has no weights**, so the held-out split constrains nothing about this comparison.

| | old | new | diff | t | same sign |
|---|---|---|---|---|---|
| **`vp_margin`** | | | | | |
| `random` | −217.8 | −237.0 | **−19.2** | −6.33 | 31/45 |
| `squad_march_take` | −1.6 | −7.5 | −5.9 | −1.50 | 29/45 |
| `squad_march_shoot` | −7.8 | −10.3 | −2.5 | −0.47 | 26/45 |
| `squad_march_deny` | −7.4 | −10.7 | −3.3 | −0.80 | 24/45 |
| **`held`** | | | | | |
| `random` | 0.11 | 0.00 | **−0.11** | −5.29 | **45/45** |
| `squad_march_take` | 2.64 | 2.37 | **−0.27** | −5.08 | 34/45 |
| `squad_march_shoot` | 2.62 | 2.41 | **−0.21** | −3.32 | 34/45 |
| `squad_march_deny` | 2.36 | 2.28 | **−0.08** | −2.01 | 29/45 |
| **`coherent`** | | | | | |
| all four | 0.16–0.81 | 0.17–0.81 | **≤ 0.01** | — | — |

**The tables are harder to hold, and that is the whole of it.** `held` falls for
every policy and `on_obj` with it (−0.02 to −0.05, t up to −5.3) — resolved, and
consistent in sign on 29 to 45 of 45.

**No script's `vp_margin` moved measurably.** Every one is inside 1.5 standard
errors, and two of the four are barely better than a coin flip on sign. The
mechanism is straightforward once stated: `vp_margin` is a *difference*, both
sides face the same harder board, and a symmetric loss of occupancy cancels.
`random` is the exception because it never contests anything — it scored by
deploying onto home objectives and standing there, which is precisely the
behaviour the change punishes, and nothing on the other side loses with it.

**Coherency is untouched**, as it should be — this changed terrain and
objectives, not formation. Two of the four differ from zero at n=45, by 0.01.

### The pool has a resolution floor, and it is about 8 vp

Per-table `vp_margin` sd is **26 to 35**, so at n=45 the standard error is
**3.9 to 5.3**. Effects under ~8 vp are not resolvable on this pool *at all* —
and more episodes per table cannot help, because the variance is **across
tables** and there are only 45 tables in existence.

**Pairing does not rescue it either, and the reason is worth keeping.** Pairing
cancels variance when the unit is identical and only the treatment differs. Here
**the table is the treatment**: `table_05` old and `table_05` new are different
boards, so pairing removes only the part of the layout that did not change. This
is the opposite of pairing two training arms on one map, where `seed_everything`
gives identical initial weights and pairing is worth an order of magnitude.

An interim read of this same comparison on the **held-out nine** made `deny` look
like the biggest mover (+4.9 → −6.6). At n=45 its old score is −7.4, not +4.9 —
the nine-table subset simply was not representative of that policy. Nine tables
is not enough to characterise a map change.

### The new bar

Held-out nine, n=30: `random` −238.8 · `take` −2.9 · `shoot` −6.9 · `deny` −6.6.
All 45, n=30: `random` −237.0 · `take` −7.5 · `shoot` −10.3 · `deny` −10.7.
Coherency 0.76–0.81 throughout.

## 7. Three bugs this turned up, two of them mine

- **Double rounding in the generator.** `round(v, 2)` and `f"{v:.2f}"` disagree at
  33.455, so an objective's outline differed by a hundredth from the very piece it
  *is*. Caught by an existing test, not by inspection. Round once.
- **Two tests had gone vacuous.** Every table now has exactly 16 pieces against a
  budget of 16, so the terrain-padding assertions were checking an empty slice
  and passing. They now trim a layout to 15.
- **A claim of mine that did not survive its own evidence.** I wrote that markers
  merge, citing "5 on 39 layouts, 4 on four, 3 on two". That histogram was
  counting *open-ground* markers, not merges. No merging occurs at all.

The new ingest tests were sensitivity-checked by injecting three bugs — a shifted
frame, a doubled control range, a removed dedupe — and confirming the right test
caught each.

## 8. What shipped

`scripts/fetch_map_layouts.py` and `just fetch-maps`, the 45 regenerated tables
and their previews, the nine synced held-out copies, and twelve tests.

**No API response is written to disk.** The layout slugs, deployment names and
mission-pack ids are the commercial product's vocabulary, and
`tests/test_no_ip_references.py` scans tracked *and untracked* files; only
geometry crosses the boundary.

Deployment zones — the other half of the source, six real shapes per layout
including one that faces the armies across the *short* axis — are deliberately
**not** in this change. They are a second scenario change and are measured
separately, or neither would be attributable.
